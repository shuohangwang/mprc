

local rankNet, parent = torch.class('mprc.rankNet', 'nn.Module')

function rankNet:__init(config)
    parent.__init(self)
    self.in_dim   = config.in_dim   or config.mem_dim
    self.mem_dim  = config.mem_dim  or 150
    self.dropoutP = config.dropoutP or 0
    self.batch_size = config.batch_size or 30
    self.lstm_layers = config.lstm_layers or 1
    self.gradInput = {self.gradInput, self.gradInput.new(), self.gradInput.new()}


    self.start_view_module  = nn.View(self.batch_size, -1)
    self.prob_module = self:new_prob_module()
    self.modules = nn.Sequential():add(self.prob_module):add(self.start_view_module)

    --self.end_pred_module = self:new_end_pred_module()
end
function rankNet:new_prob_module()
    local H, p_sizes = nn.Identity()(), nn.Identity()()
    --local wH = nn.Tanh()( nn.BLinear(self.in_dim, self.mem_dim * 2)( nn.Dropout(self.dropoutP)(H) ) )
    local maxPoolRep = nn.Max(2)(  nn.Dropout(self.dropoutP)(nn.MaskedSub(){ cudnn.BLSTM(self.in_dim, self.mem_dim / 2, self.lstm_layers, true)(nn.Dropout(self.dropoutP)(H)), p_sizes }) )
    local wH = nn.Tanh()( nn.BLinear(self.mem_dim, self.mem_dim)( maxPoolRep ))
    local prob = nn.BLinear(self.mem_dim, 1)(wH)
    local output = nn.LogSoftMax()( self.start_view_module( prob ) )
    local module = nn.gModule({H, p_sizes}, {output})
    return module
end

function rankNet:forward(inputs)
    local H, index, p_sizes = unpack(inputs)

    self.output = self.prob_module:forward({H, p_sizes})
    return self.output
end

function rankNet:backward(inputs, grad)
    local H, index, p_sizes = unpack(inputs)

    self.gradInput[2]:resize(index:size()):zero()
    self.gradInput[3]:resize(p_sizes:size()):zero()

    self.gradInput[1] = self.prob_module:backward({H, p_sizes}, grad)[1]

    return self.gradInput
end

function rankNet:share(rankNet, ...)
    if self.in_dim ~= rankNet.in_dim then error("rankNet input dimension mismatch") end
    if self.mem_dim ~= rankNet.mem_dim then error("rankNet memory dimension mismatch") end
    share_params(self.modules, rankNet.modules, ...)
end

function rankNet:zeroGradParameters()
    self.modules:zeroGradParameters()
end

function rankNet:parameters()
    return self.modules:parameters()
end

function rankNet:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end
