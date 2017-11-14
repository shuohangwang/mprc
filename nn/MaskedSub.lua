
local MaskedSub, parent = torch.class('nn.MaskedSub', 'nn.Module')

function MaskedSub:__init(val)
    parent.__init(self)
    self.defaultVal = val or 0
    self.gradInput = {self.gradInput, self.gradInput.new()}
    self.soft_modules = {}
end

function MaskedSub:updateOutput(input)
    local t = input[1]
    local sizes = input[2]
    if t:size(1) ~= sizes:size(1) then print(t:size()) print(sizes:size()) end
    assert(t:size(1) == sizes:size(1))
    self.output:resizeAs(t):fill(self.defaultVal)
    for i = 1, t:size(1) do
        if sizes[i] > 0 then
            self.output[i]:sub(1, sizes[i]):copy( t[i]:sub(1, sizes[i]) )
        end
    end
    return self.output
end

function MaskedSub:updateGradInput(input, gradOutput)
    local t = input[1]
    local sizes = input[2]

    self.gradInput[1]:resizeAs(t):zero()
    self.gradInput[2]:resize(sizes:size()):zero()
    for i = 1, t:size(1) do
        if sizes[i] > 0 then
            self.gradInput[1][i]:sub(1, sizes[i]):copy(gradOutput[i]:sub(1, sizes[i]))
        end
    end
    return self.gradInput
end

function MaskedSub:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
