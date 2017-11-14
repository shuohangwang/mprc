local BLinear, parent = torch.class('nn.BLinear', 'nn.Module')

function BLinear:__init(inputSize, outputSize, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self:reset()
end

function BLinear:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function BLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function BLinear:updateOutput(input)
   local input_sizes = input:size()
   local cum_size = 1
   for i = 1, input_sizes:size(1)-1 do cum_size = cum_size * input_sizes[i] end
   input:resize(cum_size, input_sizes[input_sizes:size(1)])
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      updateAddBuffer(self, input)
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end
   input:resize(input_sizes)
   input_sizes[input_sizes:size(1)] = self.weight:size(1)
   self.output:resize(input_sizes)

   return self.output
end

function BLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local input_sizes = input:size()
      local cum_size = 1
      for i = 1, input_sizes:size(1)-1 do cum_size = cum_size * input_sizes[i] end
      input:resize(cum_size, input_sizes[input_sizes:size(1)])
      gradOutput:resize(cum_size, self.weight:size(1))
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
      input:resize(input_sizes)
      self.gradInput:resize(input_sizes)
      gradOutput:resize(input_sizes)
      return self.gradInput
   end
end

function BLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local input_sizes = input:size()
   local cum_size = 1
   for i = 1, input_sizes:size(1)-1 do cum_size = cum_size * input_sizes[i] end
   gradOutput:resize(cum_size, self.weight:size(1))
   input:resize(cum_size, input_sizes[input_sizes:size(1)])
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   else
      error('input must be vector or matrix')
   end
   input:resize(input_sizes)
   gradOutput:resize(input_sizes)
end

-- we do not need to accumulate parameters when sharing
BLinear.sharedAccUpdateGradParameters = BLinear.accUpdateGradParameters

function BLinear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function BLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
