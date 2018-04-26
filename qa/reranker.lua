--
local reranker = torch.class('mprc.reranker')

function reranker:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.conv_dim      = config.conv_dim      or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.res_layers    = config.res_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'lstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'quasart'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0.3
    self.grad          = config.grad          or 'adamax'
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0
    self.emb_partial   = config.emb_partial   or true
    self.best_res      = 0
    self.window_sizes  = {3}
    self.pred_len_lmt  = config.pred_len_lmt  or 15
    self.res_layers    = config.res_layers    or 1
	self.lstm_layers   = config.lstm_layers   or 3
    self.expIdx        = config.expIdx        or 0
    self.smooth_val    = config.smooth_val    or 0
    self.comp_type     = config.comp_type
    self.pt_net        = config.pt_net
    self.reward        = config.reward
    self.pas_num       = config.pas_num        or 10
    self.pas_num_pred  = config.pas_num_pred   or 10

    self.sent_max_len  = config.sent_max_len   or 500


    self.batch_size_pred = self.batch_size

    self.optim_state = { learningRate = self.learning_rate }

    self.p_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.q_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.a_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)

    self.p_emb_vecs.weight:copy( tr:loadVacab2Emb(self.task):float() )
    self.p_emb_vecs.weight[1]:zero()

    self.p_dropout = nn.Dropout(self.dropoutP)
    self.q_dropout = nn.Dropout(self.dropoutP)
    self.a_dropout = nn.Dropout(self.dropoutP)

    self.p_proj_model = self:new_proj_module(self.emb_dim, self.mem_dim, 1):cuda()
    self.q_proj_model = self:new_proj_module(self.emb_dim, self.mem_dim, 1):cuda()
    self.a_proj_model = self:new_proj_module(self.emb_dim, self.mem_dim, 1):cuda()

    self.match_module = self:new_match_module(self.batch_size, self.pas_num):cuda()
    self.match_pred_module = self:new_match_module(self.batch_size_pred, self.pas_num_pred):cuda()

    self.rank_module = mprc['rankNet']({in_dim = self.mem_dim*2, mem_dim = self.mem_dim, dropoutP = self.dropoutP, batch_size = self.batch_size, lstm_layers = 3}):cuda()


    self.criterion = nn.ClassNLLCriterion():cuda()

    self.modules = nn.Sequential()
        :add(self.p_proj_model)
        :add(self.match_module)
        :add(self.rank_module)


    self.modules = self.modules:cuda()

    self.params, self.grad_params = self.modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    print(self.params:size())
    utils:share_params(self.match_pred_module, self.match_module)
    utils:share_params(self.q_emb_vecs, self.p_emb_vecs)
    utils:share_params(self.a_emb_vecs, self.p_emb_vecs)
    utils:share_params(self.q_proj_model, self.p_proj_model)
    utils:share_params(self.a_proj_model, self.p_proj_model)
end


function reranker:new_proj_module(in_dim, out_dim, layers)
    local module = nn.Sequential()
                    :add( nn.ConcatTable()
                            :add(nn.Sequential():add(nn.SelectTable(1)):add( cudnn.BLSTM(in_dim, out_dim / 2, layers, true) ) )
                            :add(nn.SelectTable(2))
                        )
                    :add(nn.MaskedSub())
    return module
end


function reranker:new_match_module(batch_size, pas_num)
    local pinput, qinput, qsizes_rep, psizes, ainput = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local qinput_rep = nn.JoinTable(2){ainput, nn.View( batch_size * pas_num,-1,self.mem_dim)( nn.Contiguous()( nn.Replicate( pas_num, 2)(qinput) ) )}
    local M_q = nn.BLinear(self.mem_dim, self.mem_dim)(qinput_rep)
    local M_pq =  nn.MM(false, true){ M_q, pinput }

    local alpha = nn.MaskedSoftMax(){M_pq, psizes}

    local p_wsum =  nn.MM(){alpha, pinput}
    local sub = nn.CSubTable(){qinput_rep, p_wsum}
    local mul = nn.CMulTable(){qinput_rep, p_wsum}
    local concate = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(4*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){p_wsum, qinput_rep, sub, mul})))
    local match = nn.MaskedSub(){concate, qsizes_rep}

    local match_module = nn.gModule({pinput, qinput, qsizes_rep, psizes, ainput}, {match})

    return match_module
end

function reranker:forward(inputs)

    local p_sents_batch, p_sizes_batch, q_sents_batch, a_sents_batch, a_sizes_batch, dev_bool = unpack(inputs)
    local batch_size = #p_sizes_batch
    local pas_num = p_sizes_batch[1]:size(1)

    self.p_words_size_batch = torch.CudaTensor(batch_size)
    for j = 1, batch_size do self.p_words_size_batch[j] = p_sizes_batch[j]:sum() end

    local p_sents_H, p_sizes = unpack(utils:padBSent(p_sents_batch, p_sizes_batch, 1))
    self.p_sents_H = p_sents_H
    local p_sents = p_sents_H:view(p_sents_H:size(1)*p_sents_H:size(2), p_sents_H:size(3))

    local a_sents_H, a_sizes = unpack(utils:padBSent(a_sents_batch, a_sizes_batch, 1))
    self.a_sents_H = a_sents_H
    local a_sents = a_sents_H:view(a_sents_H:size(1)*a_sents_H:size(2), a_sents_H:size(3))


    local q_sents, q_sizes = unpack(utils:padSent(q_sents_batch, 1))
    self.q_sizes = q_sizes:cuda()
    self.p_sizes = p_sizes:cuda()
    self.a_sizes = a_sizes:cuda()

    self.q_sizes_rep = torch.repeatTensor(q_sizes:view(-1, 1), 1,pas_num):view(-1) + a_sents_H:size(3)
    self.p_sents = p_sents
    self.q_sents = q_sents
    self.a_sents = a_sents

    self.p_inputs_emb = self.p_emb_vecs:forward(self.p_sents)
    self.q_inputs_emb = self.q_emb_vecs:forward(self.q_sents)
    self.a_inputs_emb = self.a_emb_vecs:forward(self.a_sents)

    self.p_inputs = self.p_dropout:forward(self.p_inputs_emb):cuda()
    self.q_inputs = self.q_dropout:forward(self.q_inputs_emb):cuda()
    self.a_inputs = self.a_dropout:forward(self.a_inputs_emb):cuda()

    self.p_proj = self.p_proj_model:forward({self.p_inputs, p_sizes})
    self.q_proj = self.q_proj_model:forward({self.q_inputs, q_sizes})
    self.a_proj = self.a_proj_model:forward({self.a_inputs, a_sizes})
    if not dev_bool then
        self.match_out = self.match_module:forward{self.p_proj, self.q_proj, self.q_sizes_rep, self.p_sizes, self.a_proj}
    else
        self.match_out = self.match_pred_module:forward{self.p_proj, self.q_proj, self.q_sizes_rep, self.p_sizes, self.a_proj}
    end
    --local rank_out = self.rank_module:forward({self.match_out, torch.LongTensor(1), torch.cat(p_sizes_batch)})
    local rank_out = self.rank_module:forward({self.match_out, torch.LongTensor(1), self.q_sizes_rep})

    return rank_out
end

function reranker:backward(inputs, grads)
    local p_sents_batch, p_sizes_batch, q_sents_batch = unpack(inputs)

    --local rank_grad = self.rank_module:backward({self.match_out, torch.LongTensor(1), torch.cat(p_sizes_batch)}, grads)[1]
    local rank_grad = self.rank_module:backward({self.match_out, torch.LongTensor(1), self.q_sizes_rep}, grads)[1]

    local match_grad = self.match_module:backward({self.p_proj, self.q_proj, self.q_sizes, self.p_sizes, self.a_proj}, rank_grad)

    local p_proj = self.p_proj_model:backward({self.p_inputs, self.p_sizes}, match_grad[1])
    local q_proj = self.q_proj_model:backward({self.q_inputs, self.q_sizes}, match_grad[2])
    local a_proj = self.a_proj_model:backward({self.a_inputs, self.a_sizes}, match_grad[5])

end


function reranker:batch_combine(dataset, i, indices, batch_size, max_num_cands)
    local labels = torch.LongTensor(2*batch_size)
    --local max_num_cands = self.pas_num
    local sent_max_len = self.sent_max_len

    local labels_batch = torch.LongTensor(batch_size):fill(0)
    local p_sents_batch = {}
    local q_sents_batch = {}
    local p_sizes_batch = {}
    local a_sents_batch = {}
    local a_sizes_batch = {}

    local p_words_size_batch = torch.LongTensor(batch_size)

    for j = 1, batch_size do
        local idx = i + j - 1 <= dataset.size and indices[i + j - 1] or indices[j]
        local question_tensor, passages_pn_idx, passages_pn, answers, question_id, true_id = unpack(dataset[idx])
        local passages = {}
        local p_sizes = {}
        local ans = {}
        local a_sizes = {}
        for k = 1, max_num_cands do
            if k <= passages_pn_idx:size(1) / 2 then
                local pas = passages_pn:sub(passages_pn_idx[2*k - 1], passages_pn_idx[2*k])
                passages[k] = pas:size(1) <= sent_max_len and pas or pas:sub(1, sent_max_len)
                p_sizes[k] = passages[k]:size(1)
                ans[k] = answers[k][2]
                a_sizes[k] = answers[k][2]:size(1)
            else
                passages[k] = torch.LongTensor{1}
                p_sizes[k] = 1
                ans[k] = torch.LongTensor{1}
                a_sizes[k] = 1
            end
        end

        if true_id~=nil and true_id > max_num_cands then
            local pas = passages_pn:sub(passages_pn_idx[2*true_id - 1], passages_pn_idx[2*true_id])
            passages[max_num_cands] = pas:size(1) <= sent_max_len and pas or pas:sub(1, sent_max_len)
            p_sizes[max_num_cands] = passages[max_num_cands]:size(1)

            ans[max_num_cands] = answers[true_id][2]
            a_sizes[max_num_cands] = answers[true_id][2]:size(1)

            true_id = max_num_cands
        end

        p_sizes = torch.LongTensor( p_sizes )
        local p_sents = torch.LongTensor( max_num_cands, p_sizes:max() ):fill(1)
        for k = 1, max_num_cands do
            p_sents[k]:sub(1, p_sizes[k]):copy( passages[k] )
        end

        a_sizes = torch.LongTensor( a_sizes )
        local a_sents = torch.LongTensor( max_num_cands, a_sizes:max() ):fill(1)
        for k = 1, max_num_cands do
            a_sents[k]:sub(1, a_sizes[k]):copy( ans[k] )
        end

        q_sents_batch[j] = question_tensor

        p_sents_batch[j] = p_sents
        p_sizes_batch[j] = p_sizes

        a_sents_batch[j] = a_sents
        a_sizes_batch[j] = a_sizes

        if true_id ~= nil then
            labels_batch[j] = true_id
        end
        p_words_size_batch[j] = p_sizes:sum()

    end
    return { p_sents_batch, p_sizes_batch, q_sents_batch, labels_batch, a_sents_batch, a_sizes_batch}

end

function reranker:train(dataset_inf)

    self.p_dropout:training()
    self.q_dropout:training()
    self.a_dropout:training()
    self.match_module:training()
    self.rank_module.modules:training()

    local xlua_idx = 1
    local file_num, all_size = unpack(dataset_inf)

    for file_idx = 1, file_num do
        local dataset = torch.load('../data/'..self.task..'/sequence/train'..file_idx..'.t7')
        dataset.size = #dataset
        local indices = torch.randperm(dataset.size)

        for i = 1, dataset.size, self.batch_size do

            xlua.progress(xlua_idx, all_size)
            xlua_idx = i + self.batch_size - 1 <= dataset.size and xlua_idx + self.batch_size or xlua_idx + dataset.size - i + 1
            local loss = 0
            local feval = function(x)
                self.grad_params:zero()
                self.p_emb_vecs.weight:sub(1,2):zero()

                local p_sents_batch, p_sizes_batch, q_sents_batch, labels, a_sents_batch, a_sizes_batch = unpack( self:batch_combine(dataset, i, indices, self.batch_size, self.pas_num) )

                local rank_out = self:forward( {p_sents_batch, p_sizes_batch, q_sents_batch, a_sents_batch, a_sizes_batch})

                labels = labels:cuda()
                local loss1 = self.criterion:forward(rank_out, labels)

                loss = loss1

                local crt_grad = self.criterion:backward(rank_out, labels)

                self:backward({p_sents_batch, p_sizes_batch, q_sents_batch}, crt_grad)

                self.grad_params:div(self.batch_size)

                return loss, self.grad_params
            end

            optim[self.grad](feval, self.params, self.optim_state)
            if (i - 1) / self.batch_size % 100 == 0 then  collectgarbage() end

        end
        collectgarbage()
    end
    xlua.progress(all_size, all_size)
end


function reranker:predict_dataset(dataset, set_name)
    set_name = set_name or 'dev'

    self.p_dropout:evaluate()
    self.q_dropout:evaluate()
    self.a_dropout:evaluate()
    self.match_module:evaluate()
    self.match_pred_module:evaluate()
    self.rank_module.modules:evaluate()

    local ivocab_all = torch.load('../data/'..self.task..'/ivocab_all.t7')
    local fileL = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx, "w")
    local fileL_top = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output_top.txt'..self.expIdx, "w")
    dataset.size = #dataset--/300
    --dataset.size = 300
    local indices = torch.range(1, dataset.size)
    local max_word_num = self.sent_max_len
    for i = 1, dataset.size, self.batch_size do
        self.p_emb_vecs.weight:sub(1,2):zero()
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1,  dataset.size) - i + 1

        local p_sents_batch, p_sizes_batch, q_sents_batch, _, a_sents_batch, a_sizes_batch = unpack( self:batch_combine(dataset, i, indices, self.batch_size, self.pas_num_pred) )


        local rank_out = self:forward( {p_sents_batch, p_sizes_batch, q_sents_batch, a_sents_batch, a_sizes_batch, true})
        local _, rank_id = torch.sort(rank_out, 2, true)

        for j = 1, batch_size do
            fileL_top:write(dataset[i+j-1][5])
            for k = 1, self.pas_num_pred do
                local id = rank_id[j][k] <= dataset[i+j-1][2]:size(1) / 2 and rank_id[j][k] or 1
                if k == 1 then fileL:write(dataset[i+j-1][5] .. '\t' .. dataset[i+j-1][4][ id ][1] .. '\n') end
                fileL_top:write('\t' .. dataset[i+j-1][4][ id ][1] )
            end
            fileL_top:write('\n')
        end
    end
    xlua.progress(dataset.size, dataset.size)

    local res
    if self.task == 'unftriviaqaans' then
        res = sys.execute('python ../trainedmodel/evaluation/'..self.task..'/triviaqa_evaluation.py --dataset_file ../data/'..self.task..'/raw/triviaqa-unfiltered/unfiltered-web-'..set_name..'.json --prediction_file ../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx)
    else
        res = sys.execute('python ../trainedmodel/evaluation/'..self.task..'/evaluate-v1.1.py ../data/'..self.task..'/sequence/'..set_name..'_testing.txt ../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx)
    end

    fileL:close()
    fileL_top:close()
    collectgarbage()
    return res
end

function reranker:save(path, config, result, epoch)
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
    dataname = {'dev:', 'test:'}
    for i, val in pairs(result) do
        print(dataname[i])
        print(val)

        file:write(dataname[i] .. ': ' .. val .. ', ')
    end
    file:write('\n')

    file:close()

    torch.save(paraPath, {
        params = self.params,
        config = config,
    })

    local res = stringx.split(stringx.split(result[1], ',')[1],' ')[2]
    res = tonumber(res)
    print(res)
    if res > self.best_res then
        self.best_res = res
        self.best_params:copy(self.params)
        torch.save(paraBestPath, {
            params = self.params,
            config = config
        })
    end
end

function reranker:load(path)
  local state = torch.load(path)
  --self:__init(state.config)
  self.params:copy(state.params)
end
