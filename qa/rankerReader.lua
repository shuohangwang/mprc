
local rankerReader = torch.class('mprc.rankerReader')

function rankerReader:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 30
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'quasart'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0.3
    self.grad          = config.grad          or 'adamax'
    self.pred_len_lmt  = config.pred_len_lmt  or 15
	self.lstm_layers   = config.lstm_layers   or 3
    self.expIdx        = config.expIdx        or 0
    self.smooth_val    = config.smooth_val    or 0
    self.comp_type     = config.comp_type
    self.pt_net        = config.pt_net
    self.reward        = config.reward
    self.pas_num       = config.pas_num        or 20
    self.pas_num_pred  = 50
    self.batch_size_pred = 10
    self.top_num = 50
    self.best_res = -9999

    self.optim_state = { learningRate = self.learning_rate }

    self.p_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.q_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.p_emb_vecs.weight:copy( tr:loadVacab2Emb(self.task):float() )
    self.p_emb_vecs.weight[1]:zero()

    self.p_dropout = nn.Dropout(self.dropoutP)
    self.q_dropout = nn.Dropout(self.dropoutP)

    self.p_proj_model = self:new_proj_module(self.emb_dim, self.mem_dim, 1):cuda()
    self.q_proj_model = self:new_proj_module(self.emb_dim, self.mem_dim, 1):cuda()

    self.match_module = self:new_match_module(self.batch_size, self.pas_num):cuda()
    self.match_pred_module = self:new_match_module(self.batch_size_pred, self.pas_num_pred):cuda()
    self.mlstm_module  = self:new_proj_module(self.mem_dim*2, self.mem_dim, 3)

    self.span_module = mprc[self.pt_net]({in_dim = self.mem_dim, mem_dim = self.mem_dim, dropoutP = self.dropoutP, batch_size = self.batch_size}):cuda()

    self.rank_module = mprc['rankNet']({in_dim = self.mem_dim*2, mem_dim = self.mem_dim, dropoutP = self.dropoutP, batch_size = self.batch_size}):cuda()


    self.criterion = nn.ClassNLLCriterion():cuda()

    self.modules = nn.Sequential()
        :add(self.p_proj_model)
        :add(self.match_module)
        :add(self.mlstm_module)
        :add(self.span_module)
        :add(self.rank_module)


    self.modules = self.modules:cuda()

    self.params, self.grad_params = self.modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    print(self.params:size())
    utils:share_params(self.match_pred_module, self.match_module)
    utils:share_params(self.q_emb_vecs, self.p_emb_vecs)
    utils:share_params(self.q_proj_model, self.p_proj_model)

end


function rankerReader:new_proj_module(in_dim, out_dim, layers)
    local module = nn.Sequential()
                    :add( nn.ConcatTable()
                            :add(nn.Sequential():add(nn.SelectTable(1)):add( cudnn.BLSTM(in_dim, out_dim / 2, layers, true) ) )
                            :add(nn.SelectTable(2))
                        )
                    :add(nn.MaskedSub())
    return module
end

function rankerReader:new_match_module(batch_size, pas_num)
    local pinput, qinput, qsizes_rep, psizes = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local qinput_rep = nn.View( batch_size * pas_num,-1,self.mem_dim)( nn.Contiguous()( nn.Replicate( pas_num, 2)(qinput) ) )
    local M_q = nn.View( batch_size * pas_num,-1,self.mem_dim)( nn.Contiguous()( nn.Replicate( pas_num, 2)( nn.BLinear(self.mem_dim, self.mem_dim)(qinput) ) ) )
    local M_pq = nn.MM(false, true){pinput, M_q}

    local alpha = nn.MaskedSoftMax(){M_pq, qsizes_rep}

    local q_wsum =  nn.MM(){alpha, qinput_rep}
    local sub = nn.CSubTable(){pinput, q_wsum}
    local mul = nn.CMulTable(){pinput, q_wsum}
    local concate = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(4*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){pinput, q_wsum, sub, mul})))
    local match = nn.MaskedSub(){concate, psizes}

    local match_module = nn.gModule({pinput, qinput, qsizes_rep, psizes}, {match})

    return match_module
end


function rankerReader:forward(inputs)

    local p_sents_batch, p_sizes_batch, q_sents_batch, dev_bool = unpack(inputs)
    local batch_size = #p_sizes_batch
    local pas_num = p_sizes_batch[1]:size(1)

    self.p_words_size_batch = torch.CudaTensor(batch_size)
    for j = 1, batch_size do self.p_words_size_batch[j] = p_sizes_batch[j]:sum() end

    local p_sents_H, p_sizes = unpack(utils:padBSent(p_sents_batch, p_sizes_batch, 1))
    self.p_sents_H = p_sents_H

    local p_sents = p_sents_H:view(p_sents_H:size(1)*p_sents_H:size(2), p_sents_H:size(3))

    local q_sents, q_sizes = unpack(utils:padSent(q_sents_batch, 1))
    self.q_sizes = q_sizes:cuda()
    self.p_sizes = p_sizes:cuda()
    self.q_sizes_rep = torch.repeatTensor(q_sizes:view(-1, 1), 1,pas_num):view(-1)
    self.p_sents = p_sents
    self.q_sents = q_sents

    self.p_inputs_emb = self.p_emb_vecs:forward(self.p_sents)
    self.q_inputs_emb = self.q_emb_vecs:forward(self.q_sents)

    self.p_inputs = self.p_dropout:forward(self.p_inputs_emb):cuda()
    self.q_inputs = self.q_dropout:forward(self.q_inputs_emb):cuda()

    self.p_proj = self.p_proj_model:forward({self.p_inputs, p_sizes})
    self.q_proj = self.q_proj_model:forward({self.q_inputs, q_sizes})

    if not dev_bool then
        self.match_out = self.match_module:forward{self.p_proj, self.q_proj, self.q_sizes_rep, self.p_sizes}
    else
        self.match_out = self.match_pred_module:forward{self.p_proj, self.q_proj, self.q_sizes_rep, self.p_sizes}
    end

    self.mlstm_out = self.mlstm_module:forward({self.match_out, self.p_sizes})

    self.span_in = utils:hier2flat(self.mlstm_out, p_sizes_batch)

    local span_out = self.span_module:forward({self.span_in, torch.LongTensor(1), self.p_words_size_batch})

    local span_rank_out = self.rank_module:forward({self.match_out, torch.LongTensor(1), torch.cat(p_sizes_batch)})
    assert(span_out:size(2) == self.p_words_size_batch:max())

    return {span_out, span_rank_out}
end

function rankerReader:backward(inputs, grads)
    local p_sents_batch, p_sizes_batch, q_sents_batch = unpack(inputs)
    local crt_grad, reward_grad = unpack(grads)

    local rank_grad = self.rank_module:backward({self.match_out, torch.LongTensor(1), torch.cat(p_sizes_batch)}, reward_grad)[1]

    local span_grad = self:reSpanNetBack({p_sizes_batch, self.labels_batch, self.pas_idx_batch}, crt_grad)

    local span_in_grad = utils:flat2hier(span_grad[1], p_sizes_batch)

    local layers_span_grad = {}
    local match_q_grad = torch.zeros(self.q_proj:size()):cuda()
    local mlstm_grad = self.mlstm_module:backward({self.match_out, self.p_sizes}, span_in_grad)
    mlstm_grad[1]:add( rank_grad )
    local match_grad = self.match_module:backward({self.p_proj, self.q_proj, self.q_sizes, self.p_sizes}, mlstm_grad[1])

    local p_proj = self.p_proj_model:backward({self.p_inputs, self.p_sizes}, match_grad[1])
    local q_proj = self.q_proj_model:backward({self.q_inputs, self.q_sizes}, match_grad[2])

end



function rankerReader:reSpanNetFwd(inputs)
    local p_sizes_batch, labels_batch, pas_idx_batch = unpack(inputs)
    local batch_size = #p_sizes_batch
    self.re_p_words_size_batch = torch.LongTensor(batch_size):zero()

    for i = 1, batch_size do
        self.re_p_words_size_batch[i] = p_sizes_batch[i][pas_idx_batch[i]]
        if labels_batch[i]:size(1) < self.pas_num then
            self.re_p_words_size_batch[i] = self.re_p_words_size_batch[i] + p_sizes_batch[i]:sub(labels_batch[i]:size(1)+1, -1):sum()
        end
    end

    self.re_span_in = self.span_in.new(self.batch_size, self.re_p_words_size_batch:max(), self.mem_dim):zero()
    for i = 1, batch_size do
        local cumsum_sizes = p_sizes_batch[i]:cumsum()
        local pas_idx = pas_idx_batch[i]
        self.re_span_in[i]:sub(1, p_sizes_batch[i][pas_idx]):copy( self.span_in[i]:sub(cumsum_sizes[pas_idx] - p_sizes_batch[i][pas_idx] + 1, cumsum_sizes[pas_idx]) )
        if labels_batch[i]:size(1) < self.pas_num then
            self.re_span_in[i]:sub(p_sizes_batch[i][pas_idx] + 1, p_sizes_batch[i][pas_idx] + cumsum_sizes[-1] - cumsum_sizes[labels_batch[i]:size(1)]):copy( self.span_in[i]:sub(cumsum_sizes[ labels_batch[i]:size(1) ]+1, cumsum_sizes[-1]) )
        end
    end
    local re_span_out = self.span_module:forward({self.re_span_in, torch.LongTensor(1), self.re_p_words_size_batch})
    return re_span_out
end

function rankerReader:reSpanNetBack(inputs, grad)
    local p_sizes_batch, labels_batch, pas_idx_batch = unpack(inputs)
    local batch_size = #p_sizes_batch
    local re_span_grad = self.span_module:backward({self.re_span_in, torch.LongTensor(1), self.re_p_words_size_batch}, grad)[1]

    local span_in_grad = self.span_in.new(self.span_in:size()):zero()
    for i = 1, batch_size do
        local cumsum_sizes = p_sizes_batch[i]:cumsum()
        local pas_idx = pas_idx_batch[i]

        span_in_grad[i]:sub(cumsum_sizes[pas_idx] - p_sizes_batch[i][pas_idx] + 1, cumsum_sizes[pas_idx]):copy( re_span_grad[i]:sub(1, p_sizes_batch[i][pas_idx]) )
        if labels_batch[i]:size(1) < self.pas_num then
            span_in_grad[i]:sub(cumsum_sizes[ labels_batch[i]:size(1) ]+1, cumsum_sizes[-1]):copy(
            re_span_grad[i]:sub(p_sizes_batch[i][pas_idx] + 1, p_sizes_batch[i][pas_idx] + cumsum_sizes[-1] - cumsum_sizes[ labels_batch[i]:size(1) ]) )
        end
    end
    return {span_in_grad}
end

function rankerReader:train(dataset)

    self.p_dropout:training()
    self.q_dropout:training()
    self.match_module:training()
    self.mlstm_module:training()
    self.span_module.modules:training()
    self.rank_module.modules:training()


    local datasets_num, total_size = unpack(dataset)
    local actual_size = 0
    local xlua_idx = 1
    sample.pos_pas_num = 0
    for dataset_id = 1, datasets_num do
        local dataset = torch.load('../data/' .. self.task .. '/sequence/train'..dataset_id..'.t7')
        dataset.size = #dataset
        actual_size = actual_size + dataset.size
        local indices = torch.randperm(dataset.size)
        local zeros = torch.zeros(self.mem_dim)
        for i = 1, dataset.size, self.batch_size do

            xlua.progress(xlua_idx, total_size)
            xlua_idx = i + self.batch_size - 1 <= dataset.size and xlua_idx + self.batch_size or xlua_idx + dataset.size - i + 1
            local loss = 0
            local feval = function(x)
                self.grad_params:zero()
                self.p_emb_vecs.weight[1]:zero()

                local p_sents_batch, p_sizes_batch, q_sents_batch, labels_batch = unpack( sample:batch_combine(dataset, i, indices, self.batch_size, 'samplePassages') )

                local baseline = torch.CudaTensor(self.batch_size):fill(0)
                local span_rank_out
                span_out, span_rank_out = unpack( self:forward({p_sents_batch, p_sizes_batch, q_sents_batch}) )

                local labels, reward_grad, rewards, pas_idx_batch = unpack( sample:sampleLabels(span_out, span_rank_out, p_sizes_batch, labels_batch, baseline, p_sents_batch) )
                self.pas_idx_batch = pas_idx_batch
                self.labels_batch = labels_batch

                local re_span_out = self:reSpanNetFwd({p_sizes_batch, self.labels_batch, self.pas_idx_batch})
                labels = labels:cuda()
                local loss1 = self.criterion:forward(re_span_out, labels)


                loss = loss1

                local crt_grad = self.criterion:backward(re_span_out, labels)

                self:backward({p_sents_batch, p_sizes_batch, q_sents_batch}, {crt_grad, reward_grad})

                self.grad_params:div(self.batch_size)
                return loss, self.grad_params
            end

            optim[self.grad](feval, self.params, self.optim_state)
            if (i - 1) / self.batch_size % 100 == 0 then  collectgarbage() end

        end
        collectgarbage()

    end

    xlua.progress(total_size, total_size)

end

function rankerReader:predict(p_sents_batch, q_sents_batch, p_sizes_batch, p_words_size_batch)

    local span_out, span_rank_out = unpack( self:forward({p_sents_batch, p_sizes_batch, q_sents_batch, true}) )
    span_out = span_out:float()
    span_out:add( -999999 * span_out:eq(0):float() )

    local batch_size = span_out:size(1) / 2
    local sent_len = span_out:size(2)

    for j = 1, self.batch_size_pred do
        local start_idx = 1
        for m = 1, p_sizes_batch[j]:size(1) do
            local end_idx = start_idx + p_sizes_batch[j][m] - 1
            if self.reward >= 0 then
                span_out[j]:sub(start_idx, end_idx):add(span_rank_out[j][m])
            end
            start_idx = end_idx + 1
        end
    end

    local answer_len_max = 15
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
    --sort_idx = sort_idx:float()
    sort_idx = sort_idx:float()
    sort_val = sort_val:float()

    local top_num = self.top_num
    local pred_label = sort_idx.new(batch_size, 2, top_num):zero()


    pred_label[{{},1,{}}] = torch.ceil(sort_idx[{{},{1, top_num}}] / answer_len_max)
    pred_label[{{},2,{}}] = (sort_idx[{{},{1, top_num}}]-1) % answer_len_max
    pred_label[{{},2,{}}]:add( pred_label[{{},1,{}}] )
    sort_val = sort_val[{{},{1, top_num}}]

    return {pred_label, sort_val, span_in, span_out, q_proj, q_sizes}
end


function rankerReader:predict_dataset(dataset, set_name)
    set_name = set_name or 'dev'

    self.p_dropout:evaluate()
    self.q_dropout:evaluate()
    self.match_module:evaluate()
    self.match_pred_module:evaluate()
    self.mlstm_module:evaluate()

    self.span_module.modules:evaluate()
    self.rank_module.modules:evaluate()

    local ivocab_all = torch.load('../data/'..self.task..'/ivocab_all.t7')
    local fileL = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx, "w")
    local fileL_top = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output_top.txt', "w")
    dataset.size = #dataset--/300
    --dataset.size = 300
    local pred_batch_size = self.batch_size_pred--self.batch_size
    self.span_module.start_view_module:resetSize(2*self.batch_size_pred, -1)
    self.rank_module.start_view_module:resetSize(self.batch_size_pred, -1)
    local max_word_num = 100
    for i = 1, dataset.size, pred_batch_size do

        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + pred_batch_size - 1,  dataset.size) - i + 1


        local p_sents_batch = {}
        local q_sents_batch = {}
        local p_sizes_batch = {}
        local p_words_size_batch = torch.LongTensor(pred_batch_size)
        local passages_all_batch = {}
        local question_id_batch = {}
        for j = 1, batch_size do
            local idx = i + j - 1
            local q_sent_raw, p_sent_raw, passages_pos_idx, passages_all, question_id = unpack(dataset[idx])
            if passages_pos_idx:size(1) / 2 < self.pas_num_pred then
                local pad_num = self.pas_num_pred - passages_pos_idx:size(1) / 2
                p_sent_raw = torch.cat(p_sent_raw, torch.LongTensor(pad_num):fill(1))
                passages_pos_idx = torch.cat(passages_pos_idx, torch.LongTensor(pad_num*2):fill(passages_pos_idx:size(1)+1))
            end

            local passages_pos_idx_2d = passages_pos_idx:view(-1, 2)
            local passages_pos_sizes = passages_pos_idx_2d[{{},2}] - passages_pos_idx_2d[{{},1}] + 1

            q_sents_batch[j] = q_sent_raw
            --print(passages_pos_sizes:size())
            local p_sizes_raw = passages_pos_sizes:sub( 1, self.pas_num_pred )

            local sent_max_len = p_sizes_raw:max() > max_word_num and max_word_num or p_sizes_raw:max()
            local p_sent_raw_tensor = torch.LongTensor(self.pas_num_pred, sent_max_len ):fill(1)
            local passages_all_list = {}

            for m = 1, self.pas_num_pred do
                if passages_pos_sizes[m] > max_word_num then
                    passages_pos_sizes[m] = max_word_num
                    passages_all_list[m] = passages_all:sub(passages_pos_idx_2d[m][1], passages_pos_idx_2d[m][1]+max_word_num-1)
                    p_sizes_raw[m] = max_word_num
                    p_sent_raw_tensor[m]:sub(1, passages_pos_sizes[m]):copy( p_sent_raw:sub(passages_pos_idx_2d[m][1], passages_pos_idx_2d[m][1]+max_word_num-1) )
                else
                    passages_all_list[m] = passages_all:sub(passages_pos_idx_2d[m][1], passages_pos_idx_2d[m][2])
                    p_sent_raw_tensor[m]:sub(1, passages_pos_sizes[m]):copy( p_sent_raw:sub(passages_pos_idx_2d[m][1], passages_pos_idx_2d[m][2]) )
                end
            end
            p_sents_batch[j] = p_sent_raw_tensor

            p_sizes_batch[j] = p_sizes_raw

            passages_all_batch[j] = torch.cat(passages_all_list)
            question_id_batch[j] = question_id
            p_words_size_batch[j] = p_sizes_raw:sum()
        end

        for j = batch_size+1, pred_batch_size do
            p_sents_batch[j] = p_sents_batch[1]
            p_sizes_batch[j] = p_sizes_batch[1]
            q_sents_batch[j] = q_sents_batch[1]
            p_words_size_batch[j] = p_words_size_batch[1]
        end
        local pred_batch_top, val_batch_top = unpack( self:predict(p_sents_batch, q_sents_batch, p_sizes_batch, p_words_size_batch) )
        local pred_batch = pred_batch_top[{{}, {}, 1}]
        for j = 1, batch_size do
            fileL:write(question_id_batch[j] .. '\t')

            local ps, pe = pred_batch[j][1], pred_batch[j][2]
            local passages_all = passages_all_batch[j]
            while ps <= pe do
                if ps > passages_all:size(1) then print(ps) break end
                fileL:write(ivocab_all[passages_all[ps] ])
                if ps ~= pe then fileL:write(' ') end
                ps = ps + 1
            end
            fileL:write('\n')
        end

        pred_batch = pred_batch_top
        for j = 1, batch_size do
            fileL_top:write(question_id_batch[j] .. '\t')
            for t = 1, self.top_num do
                local ps, pe = pred_batch[j][1][t], pred_batch[j][2][t]
                local val = val_batch_top[j][t]
                local passages_all = passages_all_batch[j]
                while ps <= pe do
                    if ps > passages_all:size(1) then print(ps) break end
                    fileL_top:write(ivocab_all[passages_all[ps] ])
                    if ps ~= pe then fileL_top:write(' ') end
                    ps = ps + 1
                end
                fileL_top:write('\t' .. val)

                if t ~= self.top_num then fileL_top:write('\t') end
            end
            fileL_top:write('\n')
        end


    end
    xlua.progress(dataset.size, dataset.size)
    fileL:close()
    fileL_top:close()
    local res
    if  self.task == 'unftriviaqa' then
        res = sys.execute('python ../trainedmodel/evaluation/'..self.task..'/triviaqa_evaluation.py --dataset_file ../data/'..self.task..'/raw/triviaqa-unfiltered/unfiltered-web-'..set_name..'.json --prediction_file ../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx)
    else
        res = sys.execute('python ../trainedmodel/evaluation/'..self.task..'/evaluate-v1.1.py ../data/'..self.task..'/sequence/'..set_name..'_testing.txt ../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx)
    end

    self.span_module.start_view_module:resetSize(2*self.batch_size, -1)
    self.rank_module.start_view_module:resetSize(self.batch_size, -1)
    collectgarbage()
    return res
end

function rankerReader:save(path, config, result, epoch)
    assert(string.sub(path,-1,-1)=='/')
    local paraPath = path .. config.task .. config.expIdx
    local paraBestPath = path .. config.task .. config.expIdx .. '_best'
    local recPath = path .. config.task .. config.expIdx ..'Record.txt'

    local file = io.open(recPath, 'a')
    if epoch == 1 then
        for name, val in pairs(config) do
            file:write(name .. '\t' .. tostring(val) ..'\n')
        end
    end

    file:write(config.task..': '..epoch..': ')
    for _, val in pairs(result) do
        print(val)
        file:write(val .. ', ')
    end
    file:write('\n')

    file:close()

    torch.save(paraPath, {
        params = self.params,
        config = config,
    })

    local res = stringx.split(stringx.split(result[1], ',')[1],' ')[2]
    res = tonumber(res)
    
    if res > self.best_res then
        self.best_res = res
        self.best_params:copy(self.params)
        torch.save(paraBestPath, {
            params = self.params,
            config = config
        })
    end
end

function rankerReader:load(path)
  local state = torch.load(path)
  --self:__init(state.config)
  self.params:copy(state.params)
end
