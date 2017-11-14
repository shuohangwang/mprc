

local Utils = torch.class('mprc.Utils')

function Utils:__init(config)
    self.config = config
end

function Utils:share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end

function Utils:longSentCut(sent, labels, len)
    local half_len = len / 2
    if sent:size(1) > len then
        if type(labels) == 'table' then
            local cut_start, cut_end
            local span_len = labels[2] - labels[1]
            if half_len - span_len > 5 then
                half_len = half_len - span_len
            else
                half_len = 5
            end
            if labels[1]-half_len > 1 then
                cut_start = labels[1]-half_len
            else
                cut_start = 1
            end
            if labels[2]+half_len > sent:size(1) then
                cut_end = sent:size(1)
            else
                cut_end = labels[2]+half_len
            end
            if cut_start ~= 1 then
                labels[1] = half_len + 1
                labels[2] = half_len + 1 + span_len
            end

            sent = sent:sub(cut_start, cut_end)
        else
            local cut_start, cut_end
            local span_len = labels[-1] - labels[1] + 1
            local add_len = len - span_len
            if add_len <= 0 then
                cut_start = labels[1]
                cut_end = labels[-1]
            else
                add_len = add_len > 100 and add_len or 100
                add_len_pre = add_len - 50

                if labels[1]-add_len_pre > 1 then
                    cut_start = labels[1]-add_len_pre
                else
                    cut_start = 1
                end

                add_len_pos = add_len - (labels[1]-cut_start+1)

                if labels[-1]+add_len_pos > sent:size(1) then
                    cut_end = sent:size(1)
                else
                    cut_end = labels[-1]+add_len_pos
                end
            end

            if cut_start ~= 1 then
                labels:add(1-cut_start)
            end

            assert(labels:min()>0)
            sent = sent:sub(cut_start, cut_end)

        end

    end

    return {sent, labels}
end

function Utils:padSent(sents, padVal)
    assert(type(sents) == 'table')
    padVal = padVal or 1
    local sizes = torch.LongTensor(#sents)
    for i, sent in pairs(sents) do
        sizes[i] = sent:size(1)
    end
    local max_size = sizes:max()
    local sents_pad = torch.LongTensor(#sents, max_size):fill(padVal)
    for i, sent in pairs(sents) do
        sents_pad[i]:sub(1, sizes[i]):copy(sents[i])
    end
    return {sents_pad, sizes}

end

function Utils:padBSent(sents, sizes_batch, padVal)
    assert(type(sents) == 'table')
    padVal = padVal or 1
    local sent_sizes = torch.LongTensor(#sents)
    local sents_num = torch.LongTensor(#sents)
    for i, sent in pairs(sents) do
        sent_sizes[i] = sent:size(2)
        sents_num[i] = sent:size(1)
    end
    local max_sent_size = sent_sizes:max()
    local max_sents_num = sents_num:max()
    local sents_pad = torch.LongTensor(#sents, max_sents_num, max_sent_size):fill(padVal)
    local sizes = torch.LongTensor(#sents*max_sents_num):fill(0)
    for i, sent in pairs(sents) do
        sents_pad[{i, {1, sents_num[i]}, {1, sent_sizes[i]}}]:copy(sent)
        sizes:sub((i-1)*max_sents_num+1, (i-1)*max_sents_num+sizes_batch[i]:size(1)):copy(sizes_batch[i])
    end

    return {sents_pad, sizes}

end

function Utils:samplePara(p_sents_batch, p_sizes_batch, para_idx_batch, labels, labels_b, sample_num)
    local p_sents_batch_new = {}
    local p_sizes_batch_new = {}

    local labels_new = labels.new(labels:size(1)):copy(labels)
    local labels_b_new = labels_b.new(labels_b:size(1)):copy(labels_b)

    local batch_size = #p_sents_batch
    for i, p_sents in pairs(p_sents_batch) do
        p_sizes = p_sizes_batch[i]

        local para_idx = para_idx_batch[i]
        para_idx = 1
        local sizes_sum = 0
        for j = 1, p_sizes:size(1) do
            sizes_sum = sizes_sum + p_sizes[j]
            if sizes_sum > labels[i] then
                para_idx = j~=1 and j - 1 or 1
                break
            end
        end

        local sample_num_new = p_sents:size(1) <= sample_num and p_sents:size(1)-1 or sample_num

        local indices = torch.randperm(p_sents:size(1)):sub(1, sample_num_new+1)

        if indices:eq(para_idx):sum() ~= 1 then indices[sample_num_new+1] = para_idx end
        indices = indices:sort()

        local p_sizes_new = torch.LongTensor(sample_num_new+1)
        for j = 1, sample_num_new+1 do
            p_sizes_new[j] = p_sizes[indices[j]]
        end
        local p_sents_new = p_sents.new(sample_num_new+1, p_sizes_new:max()):fill(1)
        if para_idx ~= 1 then
            labels_new[i] = labels[i] - p_sizes:sub(1,para_idx-1):sum()
            labels_new[batch_size+i] = labels[batch_size+i] - p_sizes:sub(1,para_idx-1):sum()
        end
        for j = 1, sample_num_new+1 do
            p_sents_new[j]:sub(1, p_sizes[indices[j]]):copy(p_sents[indices[j]]:sub(1, p_sizes[indices[j]]))
            if indices[j] < para_idx then
                labels_new[i] = labels_new[i] + p_sizes_new[j]
                labels_new[batch_size+i] = labels_new[batch_size+i] + p_sizes_new[j]
            end
        end
        labels_b_new[i] = labels_new[batch_size+i]
        labels_b_new[batch_size+i] = labels_new[i]
        p_sents_batch_new[i] = p_sents_new
        p_sizes_batch_new[i] = p_sizes_new
        assert(labels_new[i] > 0 and labels_new[batch_size+i] > 0)
    end
    return {p_sents_batch_new, p_sizes_batch_new, labels_new, labels_b_new}

end



function Utils:tensorReverse(tensor)
    local output = tensor.new(tensor:size())
    local num = tensor:size(1)
    for i = 1, num do
        output[i] = tensor[num - i + 1]
    end
    return output
end

function Utils:hier2flat(input, sizes_batch)
    local batch_size = #sizes_batch
    local p_words_size_batch = torch.LongTensor(batch_size)
    for i = 1, batch_size do p_words_size_batch[i] = sizes_batch[i]:sum() end

    local output = input.new(batch_size, p_words_size_batch:max(), input:size(3)):zero()
    local index = 0
    for j = 1, batch_size do
        local start_idx = 1
        for m = 1, sizes_batch[j]:size(1) do
            local end_idx = start_idx + sizes_batch[j][m] - 1
            output[{j,{start_idx, end_idx},{}}]:copy(input[index+m]:sub(1, sizes_batch[j][m]))

            start_idx = end_idx + 1
        end
        index = index + sizes_batch[j]:size(1)
    end
    return output
end

function Utils:flat2hier(input, sizes_batch)
    local batch_size = #sizes_batch
    local sizes_all = torch.cat(sizes_batch)
    local output = input.new( batch_size*sizes_batch[1]:size(1), sizes_all:max(), input:size(3) ):zero()
    local index = 0
    for j = 1, batch_size do
        local start_idx = 1
        for m = 1, sizes_batch[j]:size(1) do
            local end_idx = start_idx + sizes_batch[j][m] - 1
            output[index+m]:sub(1, sizes_batch[j][m]):copy(input[{j,{start_idx, end_idx},{}}])
            start_idx = end_idx + 1
        end
        index = index + sizes_batch[j]:size(1)
    end
    return output
end

function Utils:collectTopk(model, data, task, model_name)
    print('Generating data for reranking...')
    local train_dataset, dev_dataset, test_dataset = unpack(data)
    model_name = model_name or '../trainedmodel/'..task.. model.expIdx ..'_best_backup'
    model:load(model_name)
    topk_files = ''
    --[[]]
    if type(train_dataset)=='table' then
        local file_num, all_size = unpack(train_dataset)
        for i = 1, file_num do
            local dataset = torch.load('../data/'..task..'/sequence/train4test'..i..'.t7')
            model:predict_dataset( dataset, 'train4test'..i )
            topk_files = topk_files .. ' ../trainedmodel/evaluation/'..task..'/train4test' .. i .. '_output_top.txt'
        end

    else
        model:predict_dataset( dataset, 'train4test')
        topk_files = ' ../trainedmodel/evaluation/'..task..'/train4test_output_top.txt'
    end

    model:predict_dataset( dev_dataset, 'dev' )
    sys.execute('cp ../trainedmodel/evaluation/'..task..'/dev_output_top.txt ../trainedmodel/evaluation/'..task..'/dev_output_top.txt_backup')

    if test_dataset ~= nil then
        model:predict_dataset( test_dataset, 'test' )
        sys.execute('cp ../trainedmodel/evaluation/'..task..'/test_output_top.txt ../trainedmodel/evaluation/'..task..'/test_output_top.txt_backup')
    end

    sys.execute('cat '..topk_files..' >../trainedmodel/evaluation/'..task..'/train4test_top.txt')

    print('Top k collection finish!')
    os.exit()
end
