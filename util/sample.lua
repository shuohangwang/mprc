local Sample = torch.class('mprc.Sample')

function Sample:__init(config)
    self.smooth_val    = config.smooth_val    or 0
    self.batch_size    = config.batch_size
    self.reward        = config.reward        or 0
    self.pas_num       = config.pas_num       or 10
end

function Sample:getPredProb(span_out_raw, p_sizes_batch, labels_batch, answer_len_max, top_num)
    local span_out_raw_new = span_out_raw:float()
    for i = 1, #p_sizes_batch do
        local size = p_sizes_batch[i]:sum()
        if size < span_out_raw_new[i]:size(1) then
            span_out_raw_new[i]:sub(size+1,-1):add( -999999 )
        end
    end

    local sent_sizes_max = torch.cat(p_sizes_batch):max()
    local s_batch_size = span_out_raw_new:size(1) / 2
    local s_pas_num = p_sizes_batch[1]:size(1)

    local sent_size_batch = torch.zeros(s_batch_size)
    local span_out = span_out_raw_new.new(s_batch_size*2, s_pas_num, sent_sizes_max ):fill(-999999)

    for i = 1, s_batch_size do
        local start_idx = 1
        for j = 1, s_pas_num do
            local end_idx = start_idx + p_sizes_batch[i][j] - 1
            span_out[i][j]:sub(1, p_sizes_batch[i][j]):copy( span_out_raw_new[i]:sub(start_idx, end_idx) )
            span_out[s_batch_size+i][j]:sub(1, p_sizes_batch[i][j]):copy( span_out_raw_new[s_batch_size+i]:sub(start_idx, end_idx)  )

            start_idx = end_idx + 1
        end
        sent_size_batch[i] = p_sizes_batch[i]:sum()
    end
    assert(sent_size_batch:max() == span_out_raw:size(2))

    span_out:resize(s_batch_size*2*s_pas_num, sent_sizes_max)

    local batch_size = span_out:size(1) / 2
    local sent_len = span_out:size(2)

    local answer_prob_matrix = torch.repeatTensor(span_out:sub(1, batch_size):view(batch_size, -1, 1), 1, 1, answer_len_max )
    local end_prob = answer_prob_matrix.new(batch_size, answer_len_max):fill(-999999)
    for j = 1, sent_len do
        if j + answer_len_max - 1 > sent_len then
            end_prob:fill(-999999)
            end_prob[{{},{1, sent_len-j+1}}]:copy(span_out[{{batch_size+1, -1}, {j, -1}}])
        else
            end_prob:copy( span_out[{{batch_size+1, -1}, {j, j+answer_len_max-1}}] )
        end
        answer_prob_matrix[{{},{j},{}}]:add(end_prob)
    end

    answer_prob_matrix = answer_prob_matrix:view(batch_size, -1)

    local sort_val, sort_idx = answer_prob_matrix:sort(2, true)

    local top_num = 1
    local pred_label = torch.LongTensor(batch_size, 2, top_num):zero()
    sort_idx = sort_idx:float()
    sort_val = sort_val:float()

    pred_label[{{},1,{}}] = torch.ceil(sort_idx[{{},{1, top_num}}] / answer_len_max)
    pred_label[{{},2,{}}] = (sort_idx[{{},{1, top_num}}]-1) % answer_len_max

    pred_label[{{},2,{}}]:add( pred_label[{{},1,{}}] )

    sort_val = sort_val[{{},{1, top_num}}]

    return {pred_label, sort_val}
end

function Sample:sampleLabels(span_out, span_rank_out, p_sizes_batch, labels_batch, baseline, p_sents_batch)
    local pred_label,sort_val = unpack(self:getPredProb(span_out, p_sizes_batch, labels_batch, 15, 1))

    local sort_val_rank = span_rank_out

    pred_label = pred_label:view(self.batch_size, self.pas_num, 2)
    sort_val = sort_val[{{},1}]:contiguous():view(self.batch_size, self.pas_num)


    local labels = torch.LongTensor(2*self.batch_size)
    local predictions = torch.LongTensor(2*self.batch_size)

    local rewards = torch.zeros(self.batch_size)

    local answers = {}
    local reward_val = self.reward
    local reward_grad = torch.zeros(span_rank_out:size())
    local pas_idx_batch = torch.LongTensor(self.batch_size)
    local entropy = {}
    local accuracy = 0
    local reward_max = torch.zeros(self.batch_size)
    for j = 1, self.batch_size do
        local ent_vals = torch.exp(sort_val_rank[j])
        local pas_vals = ent_vals:sub(1, labels_batch[j]:size(1))+self.smooth_val

        if pas_vals:sum() <= 0 then print(pas_vals) print(p_sizes_batch[j]) pas_vals:fill(1) end

        if ent_vals:size(1) > 1 then
            local ent_vals_norm = ent_vals / ent_vals:sum()
            entropy[#entropy + 1] = - ent_vals_norm:cmul(torch.log(ent_vals_norm)):sum()
        end

        local pas_idx = torch.multinomial(pas_vals, 1)[1]

        local _, m_idx = torch.max(ent_vals, 1)

        if reward_val == 0 then
            pas_idx = torch.random(pas_vals:size(1))
        end

        if m_idx[1] <= labels_batch[j]:size(1) then
            accuracy = accuracy + 1
        end

        pas_idx_batch[j] = pas_idx

        local start = pas_idx > 1 and p_sizes_batch[j]:sub(1, pas_idx-1):sum() or 0


        labels[j] = labels_batch[j][pas_idx][1]
        labels[self.batch_size + j] = labels_batch[j][pas_idx][2]

        predictions[j] = pred_label[j][ pas_idx ][1]
        predictions[self.batch_size + j] = pred_label[j][ pas_idx ][2]

        rewards[j] = self:getReward(predictions[j], predictions[self.batch_size + j], labels[j], labels[self.batch_size + j], reward_val, p_sents_batch[j][pas_idx])


        if self.reward > 0 then
            reward_grad[j][ pas_idx ] = rewards[j]
        else
            local r_val = 1 / labels_batch[j]:size(1)
            for m = 1, labels_batch[j]:size(1) do
                reward_grad[j][ m ] = r_val
            end
        end
    end
    return {labels, -reward_grad:cuda() / self.batch_size, rewards, pas_idx_batch}
end


function Sample:getReward_backup(pred1, pred2, label1, label2, reward_val, sent)
    local stopwords = {a=1, the=1}
    stopwords[','] = 1 stopwords['.'] = 1 stopwords['!'] = 1

    local reward

    if (pred1 == label1) and (pred2 == label2) then
        reward = 2*reward_val
    else
        local pred_words = {}
        local pred_len = 0
        for j = pred1, pred2 do
            if j <= sent:size(1) then
                local w = ivocab[ sent[j] ]
                if stopwords[w] ~= nil then
                    pred_words[ ivocab[ sent[j] ] ] = 1
                    pred_len = pred_len + 1
                end
            end
        end

        if pred_len == 0 then return -1 * reward_val end

        local label_len = 0
        local match_len = 0
        for j = label1, label2 do
            if j <= sent:size(1) then
                local w = ivocab[ sent[j] ]
                if stopwords[w] ~= nil then
                    if pred_words[w] ~= nil then match_len = match_len + 1 end
                    label_len = label_len + 1
                end
            end
        end


        local g_len = match_len / label_len
        local p_len = match_len / pred_len
        if match_len == 0 then
            reward = -1 * reward_val
        else
            reward = 2.0 / ( 1.0/g_len + 1.0/p_len ) * reward_val
            if reward == reward_val then reward = 2*reward_val end
        end
    end
    return reward
end

function Sample:getReward(pred1, pred2, label1, label2, reward_val)
    --print(pred1..' '..pred2..' '..label1..' '..label2)
    local reward
    if (pred1 == label1) and (pred2 == label2) then
        reward = 2*reward_val

    elseif pred1 <= label1 and pred2 >= label1 then
        local match_len = pred2 <= label2 and  pred2 - label1 + 1.0 or label2 - label1 + 1.0
        local g_len = match_len / (label2 - label1 + 1.0)
        local p_len = match_len / (pred2 - pred1 + 1.0)
        reward = 2.0 / ( 1.0/g_len + 1.0/p_len ) * reward_val
        assert(reward <= 1)

    elseif pred1 <= label2 and pred2 >= label2 then
        local match_len = pred1 <= label1 and label2 - label1 + 1.0 or label2 - pred1 + 1.0
        local g_len = match_len / (label2 - label1 + 1.0)
        local p_len = match_len / (pred2 - pred1 + 1.0)
        reward = 2.0 / ( 1.0/g_len + 1.0/p_len ) * reward_val
        assert(reward <= 1)

    else
        reward = -1 * reward_val
    end
    return reward
end

function Sample:batch_combine(dataset, i, indices, batch_size, sample_method, sample_pos_idx)
    local labels = torch.LongTensor(2*batch_size)

    local labels_batch = {}
    local p_sents_batch = {}
    local q_sents_batch = {}
    local p_sizes_batch = {}
    local p_words_size_batch = torch.LongTensor(batch_size)

    for j = 1, self.batch_size do
        local idx = i + j - 1 <= dataset.size and indices[i + j - 1] or indices[j]

        local p_sent_raw, p_sizes_raw, q_sent_raw, labels_raw = unpack(self[sample_method](self, dataset[idx]))

        p_sents_batch[j] = p_sent_raw
        q_sents_batch[j] = q_sent_raw
        p_sizes_batch[j] = p_sizes_raw
        labels_batch[j] = labels_raw

        p_words_size_batch[j] = p_sizes_raw:sum()

    end
    return { p_sents_batch, p_sizes_batch, q_sents_batch, labels_batch}
end

function Sample:samplePassages(data)
    local question_tensor, passages_pos, passages_pos_idx, passages_neg, passages_neg_idx, passages_pts = unpack(data)
    local max_pas_len = 50
    assert(passages_pos_idx:size(1) == passages_pts:size(1))

    local pas_num = self.pas_num

    local pos_num = passages_pos_idx:size(1) / 2
    self.pos_pas_num = self.pos_pas_num + pos_num

    local pos_ids
    if pos_num >= self.pas_num*4/5 and passages_neg_idx:dim() ~= 0 and passages_neg_idx:size(1)/2 >= self.pas_num/5 then
        pos_ids = torch.randperm(pos_num):sub(1, self.pas_num*4/5)
    elseif pos_num >= self.pas_num then
        pos_ids = torch.randperm(pos_num):sub(1, self.pas_num)
    else
        pos_ids = torch.range(1, pos_num)
    end

    local labels = torch.LongTensor(pos_ids:size(1), 2)

    local passages = {}
    local sizes = torch.LongTensor(self.pas_num)

    for i = 1, pos_ids:size(1) do
        local pos_id = pos_ids[i]
        passages[i] = passages_pos:sub(passages_pos_idx[2*pos_id-1], passages_pos_idx[2*pos_id])
        local sents = utils:longSentCut(passages[i], {passages_pts[2*pos_id-1], passages_pts[2*pos_id]}, max_pas_len)
        passages[i] = sents[1]
        labels[i] = torch.LongTensor(sents[2])
        sizes[i] = passages[i]:size(1)
    end

    if pos_ids:size(1) < self.pas_num and passages_neg_idx:dim() ~= 0 then
        local neg_ids
        if self.pas_num - pos_ids:size(1) <= passages_neg_idx:size(1) / 2 then
            neg_ids = torch.randperm(passages_neg_idx:size(1) / 2):sub(1, self.pas_num - pos_ids:size(1))
        else
            neg_ids = torch.repeatTensor( torch.randperm(passages_neg_idx:size(1) / 2), self.pas_num - pos_ids:size(1) ):sub(1, self.pas_num - pos_ids:size(1))
        end
        for i = 1, neg_ids:size(1) do
            local idx = i+pos_ids:size(1)
            local neg_id = neg_ids[i]
            if passages_neg_idx[2*neg_id] - passages_neg_idx[2*neg_id-1] > max_pas_len then
                local start_idx = math.random(passages_neg_idx[2*neg_id] - passages_neg_idx[2*neg_id-1] - max_pas_len)

                passages[idx] = passages_neg:sub(passages_neg_idx[2*neg_id-1]+start_idx, passages_neg_idx[2*neg_id-1]+start_idx+max_pas_len)
            else
                passages[idx] = passages_neg:sub(passages_neg_idx[2*neg_id-1], passages_neg_idx[2*neg_id])
            end
            sizes[idx] = passages[idx]:size(1)
        end
    end

    while #passages < self.pas_num do
        passages[#passages+1] = torch.LongTensor{1}
        sizes[#passages] = 1
    end
    local max_size = sizes:max()

    assert(#passages == self.pas_num)
    local p = torch.LongTensor(pas_num, max_size):fill(1)
    for i = 1, self.pas_num do
        p[i]:sub(1, sizes[i]):copy(passages[i])
    end
    return {p, sizes, question_tensor, labels}
end
