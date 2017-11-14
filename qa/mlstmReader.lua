
local mlstmReader = torch.class('mprc.mlstmReader')

function mlstmReader:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.conv_dim      = config.conv_dim      or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.res_layers    = config.res_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'lstm'
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'paraphrase'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0
    self.emb_partial   = config.emb_partial   or true
    self.best_res      = 0
    self.window_sizes  = {3}
    self.pred_len_lmt  = config.pred_len_lmt  or 15
    self.res_layers    = config.res_layers    or 3
	self.lstm_layers   = config.lstm_layers   or 1
    self.expIdx        = config.expIdx        or 0
    self.comp_type     = config.comp_type
    self.pt_net        = config.pt_net
    self.top_num       = 50

    self.optim_state = { learningRate = self.learning_rate }

    self.p_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.q_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.p_emb_vecs.weight:copy( tr:loadVacab2Emb(self.task):float() )
    self.p_emb_vecs.weight[1]:zero()

    self.p_dropout = nn.Dropout(self.dropoutP)
    self.q_dropout = nn.Dropout(self.dropoutP)

    self.blstms = {cudnn.BLSTM(self.emb_dim, self.mem_dim/2, 1, true), cudnn.BLSTM(self.emb_dim, self.mem_dim/2, 1, true)}

    self.p_proj_model = nn.Sequential()
                        :add( nn.ConcatTable()
                                :add(nn.Sequential():add(nn.SelectTable(1)):add(self.blstms[1]) )
                                :add(nn.SelectTable(2))
                            )
                        :add(nn.MaskedSub()):cuda()
    self.q_proj_model = nn.Sequential()
                        :add( nn.ConcatTable()
                                :add(nn.Sequential():add(nn.SelectTable(1)):add(self.blstms[2]) )
                                :add(nn.SelectTable(2))
                            )
                        :add(nn.MaskedSub()):cuda()
    self.match_modules = {}
    self.mlstm_modules  = {}

    self.match_blstms = {}
    for i = 1, self.res_layers do
        self.match_blstms[i] = cudnn.BLSTM(self.mem_dim, self.mem_dim/2, 1, true)
        self.blstms[i+2] = cudnn.BLSTM(self.mem_dim*2, self.mem_dim/2, self.lstm_layers, true)
        self.match_modules[i] = self:new_match_module():cuda()
        self.mlstm_modules[i]  = nn.Sequential()
                            :add( nn.ConcatTable()
                                    :add(nn.Sequential():add(nn.SelectTable(1)):add(self.blstms[i+2]) )
                                    :add(nn.SelectTable(2))
                                )
                            :add(nn.MaskedSub()):cuda()
    end

    self.span_module = mprc[self.pt_net]({in_dim = self.mem_dim, mem_dim = self.mem_dim, dropoutP = self.dropoutP, batch_size = self.batch_size}):cuda()

    self.criterion = nn.ClassNLLCriterion():cuda()

    self.modules = nn.Sequential()
        :add(self.p_proj_model)
        :add(self.span_module)

    for i = 1, self.res_layers do
        self.modules:add(self.match_modules[i])
        self.modules:add(self.mlstm_modules[i])
    end
    self.modules = self.modules:cuda()
    self.params, self.grad_params = self.modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    print(self.params:size())
    utils:share_params(self.q_emb_vecs, self.p_emb_vecs)
    utils:share_params(self.q_proj_model, self.p_proj_model)

end


function mlstmReader:new_match_module()
    local pinput, qinput, qsizes, psizes = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local M_q = nn.BLinear(self.mem_dim, self.mem_dim)(qinput)
    local M_pq = nn.MM(false, true){pinput, M_q}

    local alpha = nn.MaskedSoftMax(){M_pq, qsizes}

    local q_wsum =  nn.MM(){alpha, qinput}
    local sub = nn.CSubTable(){pinput, q_wsum}
    local mul = nn.CMulTable(){pinput, q_wsum}
    local concate = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(4*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){pinput, q_wsum, sub, mul})))
    local match = nn.MaskedSub(){concate, psizes}

    local match_module = nn.gModule({pinput, qinput, qsizes, psizes}, {match})

    return match_module

end
function mlstmReader:train(dataset)
    self.p_dropout:training()
    self.q_dropout:training()
    for i = 1, self.res_layers do
        self.match_modules[i]:training()
        self.mlstm_modules[i]:training()
    end

    dataset.size = #dataset
    local indices = torch.randperm(dataset.size)
    local zeros = torch.zeros(self.mem_dim)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local loss = 0
        local feval = function(x)
            self.grad_params:zero()
            self.p_emb_vecs.weight[1]:zero()

            local labels = torch.LongTensor(2*self.batch_size)
            local labels_b = torch.LongTensor(2*self.batch_size)

            local p_sents_batch = {}
            local q_sents_batch = {}
            for j = 1, self.batch_size do
                local idx = i + j - 1 <= dataset.size and indices[i + j - 1] or indices[j]
                local p_sent_raw, q_sent_raw, labels_raw = unpack(dataset[idx])
                p_sent_raw, labels_raw = unpack( utils:longSentCut(p_sent_raw, {labels_raw[1], labels_raw[-2]}, 400) )
                p_sents_batch[j] = p_sent_raw
                q_sents_batch[j] = q_sent_raw
                assert(#labels_raw >= 2)
                labels[j], labels[self.batch_size+j] = labels_raw[1], labels_raw[2]
                labels_b[j], labels_b[self.batch_size+j] = labels_raw[2], labels_raw[1]
            end

            local p_sents, p_sizes = unpack(utils:padSent(p_sents_batch, 1))
            local q_sents, q_sizes = unpack(utils:padSent(q_sents_batch, 1))
            q_sizes = q_sizes:cuda()
            p_sizes = p_sizes:cuda()
            labels = labels:cuda()
            labels_b = labels_b:cuda()

            local p_inputs_emb = self.p_emb_vecs:forward(p_sents)
            local q_inputs_emb = self.q_emb_vecs:forward(q_sents)

            local p_inputs = self.p_dropout:forward(p_inputs_emb):cuda()
            local q_inputs = self.q_dropout:forward(q_inputs_emb):cuda()

            local p_proj = self.p_proj_model:forward({p_inputs, p_sizes})
            local q_proj = self.q_proj_model:forward({q_inputs, q_sizes})

            local layers_input = {}
            local match_out = {}
            for l = 1, self.res_layers do
                if l ~= 1 then
                    layers_input[l] = layers_input[l] + layers_input[l-1]
                else
                    layers_input[l] = p_proj
                end
                match_out[l] = self.match_modules[l]:forward{layers_input[l], q_proj, q_sizes, p_sizes}
                layers_input[l+1] = self.mlstm_modules[l]:forward({match_out[l], p_sizes})
            end


            local span_out = self.span_module:forward({layers_input[self.res_layers+1], labels:sub(1, self.batch_size), p_sizes})
            assert(span_out:size(2) == p_sents:size(2))
            loss = self.criterion:forward(span_out, labels)

            local crt_grad = self.criterion:backward(span_out, labels)

            local span_grad = self.span_module:backward({layers_input[self.res_layers+1], labels:sub(1, self.batch_size), p_sizes}, crt_grad)

            local layers_span_grad = {}
            local match_q_grad = torch.zeros(q_proj:size()):cuda()
            for l = self.res_layers, 1, -1 do
                if l == self.res_layers then
                    layers_span_grad[l] = span_grad[1]
                elseif l == self.res_layers - 1 then
                    layers_span_grad[l] = layers_span_grad[l]
                else
                    layers_span_grad[l] = layers_span_grad[l] + layers_span_grad[l+1]
                end
                local conv_grad = self.mlstm_modules[l]:backward({match_out[l], p_sizes}, layers_span_grad[l])
                local match_grad = self.match_modules[l]:backward({layers_input[l], q_proj, q_sizes, p_sizes}, conv_grad[1])
                layers_span_grad[l-1] = match_grad[1]
                match_q_grad:add(match_grad[2])
            end

            local p_proj = self.p_proj_model:backward({p_inputs, p_sizes}, layers_span_grad[0])
            local q_proj = self.q_proj_model:backward({q_inputs, q_sizes}, match_q_grad)

            self.grad_params:div(self.batch_size)
            return loss, self.grad_params
        end
        optim[self.grad](feval, self.params, self.optim_state)

    end
    xlua.progress(dataset.size, dataset.size)
    collectgarbage()
end


function mlstmReader:predict(p_sents_batch, q_sents_batch)
    local p_sents, p_sizes = unpack(utils:padSent(p_sents_batch, 1))
    local q_sents, q_sizes = unpack(utils:padSent(q_sents_batch, 1))
    q_sizes = q_sizes:cuda()
    p_sizes = p_sizes:cuda()

    local p_inputs_emb = self.p_emb_vecs:forward(p_sents)
    local q_inputs_emb = self.q_emb_vecs:forward(q_sents)

    local p_inputs = self.p_dropout:forward(p_inputs_emb):cuda()
    local q_inputs = self.q_dropout:forward(q_inputs_emb):cuda()

    local p_proj = self.p_proj_model:forward({p_inputs, p_sizes})
    local q_proj = self.q_proj_model:forward({q_inputs, q_sizes})

    local layers_input = {}
    local match_out = {}
    for l = 1, self.res_layers do
        if l ~= 1 then
            layers_input[l] = layers_input[l] + layers_input[l-1]
        else
            layers_input[l] = p_proj
        end
        match_out[l] = self.match_modules[l]:forward{layers_input[l], q_proj, q_sizes, p_sizes}
        layers_input[l+1] = self.mlstm_modules[l]:forward({match_out[l], p_sizes})
    end

    local span_out = self.span_module:forward({layers_input[self.res_layers+1], torch.LongTensor(1), p_sizes})

    span_out = span_out:float()
    span_out:add( -999999 * span_out:eq(0):float() )

    local batch_size = span_out:size(1) / 2
    local sent_len = span_out:size(2)

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

    sort_idx = sort_idx:float()
    sort_val = sort_val:float()

    local top_num = self.top_num
    local pred_label = sort_idx.new(batch_size, 2, top_num):zero()


    pred_label[{{},1,{}}] = torch.ceil(sort_idx[{{},{1, top_num}}] / answer_len_max)
    pred_label[{{},2,{}}] = (sort_idx[{{},{1, top_num}}]-1) % answer_len_max
    pred_label[{{},2,{}}]:add( pred_label[{{},1,{}}] )
    sort_val = sort_val[{{},{1, top_num}}]


    return {pred_label, sort_val}
end

function mlstmReader:predict_dataset(dataset, set_name)

    set_name = set_name or 'dev'
    self.p_dropout:evaluate()
    self.q_dropout:evaluate()
    for i = 1, self.res_layers do
        self.match_modules[i]:evaluate()
        self.mlstm_modules[i]:evaluate()
    end
    local ivocab = tr:loadiVocab(self.task)
    local fileL_top = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output_top.txt'..self.expIdx, "w")
    local fileL = io.open('../trainedmodel/evaluation/'..self.task..'/'..set_name..'_output.txt'..self.expIdx, "w")
    dataset.size = #dataset
    local pred_batch_size = self.batch_size
    for i = 1, dataset.size, pred_batch_size do

        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + pred_batch_size - 1,  dataset.size) - i + 1

        local p_sents_batch = {}
        local q_sents_batch = {}
        for j = 1, batch_size do
            local idx = i + j - 1
            local p_sent_raw, q_sent_raw = unpack(dataset[idx])
            p_sents_batch[j] = p_sent_raw
            q_sents_batch[j] = q_sent_raw
        end

        for j = batch_size+1, pred_batch_size do
            p_sents_batch[j] = p_sents_batch[1].new{1}
            q_sents_batch[j] = q_sents_batch[1].new{1}
        end

        local pred_batch_top, val_batch_top = unpack( self:predict(p_sents_batch, q_sents_batch) )
        local pred_batch = pred_batch_top[{{}, {}, 1}]
        for j = 1, batch_size do
            local ps, pe = pred_batch[j][1], pred_batch[j][2]
            while ps <= pe do
                if ps > p_sents_batch[j]:size(1) then print(ps) break end
                fileL:write(ivocab[p_sents_batch[j][ps] ])
                if ps ~= pe then fileL:write(' ') end
                ps = ps + 1
            end
            fileL:write('\n')
        end

        pred_batch = pred_batch_top
        for j = 1, batch_size do
            for t = 1, self.top_num do
                local ps, pe = pred_batch[j][1][t], pred_batch[j][2][t]
                local val = val_batch_top[j][t]
                while ps <= pe do
                    if ps > p_sents_batch[j]:size(1) then print(ps) break end
                    fileL_top:write(ivocab[ p_sents_batch[j][ps] ])
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
    sys.execute('python ../trainedmodel/evaluation/squad/txt2js.py ../data/squad/'..set_name..'-v1.1.json ../trainedmodel/evaluation/squad/'..set_name..'_output.txt'..self.expIdx..' ../trainedmodel/evaluation/squad/prediction.json'..self.expIdx)
    local res = sys.execute('python ../trainedmodel/evaluation/squad/evaluate-v1.1.py ../data/squad/'..set_name..'-v1.1.json ../trainedmodel/evaluation/squad/prediction.json'..self.expIdx)

    fileL:close()

    collectgarbage()
    return res
end


function mlstmReader:save(path, config, result, epoch)
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
            params = self.params:float(),
            config = config
        })
    end
end

function mlstmReader:load(path)
  local state = torch.load(path)
  self:__init(state.config)
  self.params:copy(state.params)
end
