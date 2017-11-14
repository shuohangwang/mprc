
local MaskedLogSoftMax, parent = torch.class('nn.MaskedLogSoftMax', 'nn.Module')

function MaskedLogSoftMax:__init(dimension)
    parent.__init(self)
    self.dimension = dimension
    self.gradInput = {self.gradInput, self.gradInput.new()}
    self.soft_modules = {}
end

function MaskedLogSoftMax:updateOutput(input)
    local t_raw = input[1]
    local sizes = input[2]

    assert(t_raw:dim() == 3)
    assert(t_raw:size(1) == sizes:size(1))

    local sizes_max = sizes:max()

    local t = t_raw:size(3) == sizes_max and t_raw or t_raw:view(t_raw:size(1), -1, sizes_max)

    self.output:resizeAs(t):zero()

    for i = 1, t:size(1) do
        local soft_module = self.soft_modules[i]
        if soft_module == nil then
            self.soft_modules[i] = nn.LogSoftMax():cuda()
            soft_module = self.soft_modules[i]
        end
        if sizes[i] ~= 0 then
            local output = soft_module:forward(t[{i, {}, {1, sizes[i]}}])
            self.output[{i, {}, {1, sizes[i]}}]:copy( output )
        end
    end
    return self.output
end

function MaskedLogSoftMax:updateGradInput(input, gradOutput)
    local t_raw = input[1]
    local sizes = input[2]

    local sizes_max = sizes:max()
    local t = t_raw:size(3) == sizes_max and t_raw or t_raw:view(t_raw:size(1), -1, sizes_max)

    self.gradInput[1]:resizeAs(t):zero()
    self.gradInput[2]:resize(sizes:size()):zero()
    for i = 1, t:size(1) do
        local soft_module = self.soft_modules[i]
        if sizes[i] ~= 0 then
            local grad = soft_module:backward(t[{i, {}, {1, sizes[i]}}], gradOutput[{i, {}, {1, sizes[i]}}])
            self.gradInput[1][{i, {}, {1, sizes[i]}}]:copy(grad)
        end
    end
    if t_raw:size(3) ~= sizes_max then self.gradInput[1]:resizeAs(t_raw) end

    return self.gradInput
end

function MaskedLogSoftMax:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
