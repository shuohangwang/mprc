

local spanNet, parent = torch.class('mprc.spanNet', 'nn.Module')

function spanNet:__init(config)
    parent.__init(self)
    self.in_dim   = config.in_dim   or config.mem_dim
    self.mem_dim  = config.mem_dim  or 150
    self.dropoutP = config.dropoutP or 0
    self.batch_size = config.batch_size or 30
    self.gradInput = {self.gradInput, self.gradInput.new(), self.gradInput.new()}


    self.start_view_module  = nn.View(self.batch_size*2, -1)
    self.prob_module = self:new_prob_module()
    self.modules = nn.Sequential():add(self.prob_module):add(self.start_view_module)

    --self.end_pred_module = self:new_end_pred_module()
end
function spanNet:new_prob_module()
    local H, p_sizes = nn.Identity()(), nn.Identity()()
    local wH = nn.Tanh()( nn.BLinear(self.in_dim, self.mem_dim * 2)( nn.Dropout(self.dropoutP)(H) ) )
    local start_prob = nn.Transpose({2,3})( nn.BLinear(self.mem_dim, 1)(nn.Narrow(3, 1, self.mem_dim)(wH) ) )
    local end_prob = nn.Transpose({2,3})( nn.BLinear(self.mem_dim, 1)(nn.Narrow(3, self.mem_dim+1, self.mem_dim)(wH) ) )
    local prob = nn.JoinTable(2){start_prob, end_prob}
    local soft = nn.MaskedLogSoftMax(){ prob, p_sizes }
    local output = self.start_view_module( nn.Contiguous()( nn.Transpose({1,2})(soft) ) )
    local module = nn.gModule({H, p_sizes}, {output})
    return module
end

function spanNet:forward(inputs)
    local H, index, p_sizes = unpack(inputs)
    self.output = self.prob_module:forward({H, p_sizes})
    return self.output
end

function spanNet:backward(inputs, grad)
    local H, index, p_sizes = unpack(inputs)

    self.gradInput[2]:resize(index:size()):zero()
    self.gradInput[3]:resize(p_sizes:size()):zero()
    self.gradInput[1] = self.prob_module:backward({H, p_sizes}, grad)[1]
    return self.gradInput
end

function spanNet:predict(inputs, len)
    local H, p_sizes = unpack(inputs)
    local H_sizes = H:size()
    self.start_prob = self.start_module:forward({H, p_sizes})
    self.start_prob:add( -999999*self.start_prob:eq(0):cuda() )

    self.end_prob = self.end_module:forward{H, H, p_sizes}
    self.end_prob:add( -999999*self.end_prob:eq(0):cuda() )

    for i = 1, H_sizes[1] do
        local pred_start = torch.repeatTensor(self.start_prob[i]:view(-1, 1), 1, H_sizes[2])
        self.end_prob[i]:add(pred_start)
    end

    return self.end_prob
end

function spanNet:share(spanNet, ...)
    if self.in_dim ~= spanNet.in_dim then error("spanNet input dimension mismatch") end
    if self.mem_dim ~= spanNet.mem_dim then error("spanNet memory dimension mismatch") end
    share_params(self.modules, spanNet.modules, ...)
end

function spanNet:zeroGradParameters()
    self.modules:zeroGradParameters()
end

function spanNet:parameters()
    return self.modules:parameters()
end

function spanNet:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end
