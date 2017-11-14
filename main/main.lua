
require 'nngraph'
require 'optim'
require 'debug'

torch.setdefaulttensortype('torch.FloatTensor')
mprc = {}
tr = require '../util/loadFiles'
include '../util/utils.lua'
include '../util/sample.lua'


include '../qa/mlstmReader.lua'
include '../qa/rankerReader.lua'

include '../models/spanNet.lua'
include '../models/rankNet.lua'

include '../nn/BLinear.lua'
include '../nn/MaskedSoftMax.lua'
include '../nn/MaskedLogSoftMax.lua'
include '../nn/MaskedSub.lua'

print ("require done !")

cmd = torch.CmdLine()
cmd:option('-batch_size',30,'number of sequences to train on in parallel')
cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-seed',2345,'torch manual random number generator seed')
cmd:option('-reg',0,'regularize value')
cmd:option('-learning_rate',0.002,'learning rate')
cmd:option('-lr_decay',0.95,'learning rate decay ratio')
cmd:option('-dropoutP',0.3,'dropout ratio')
cmd:option('-expIdx', 1, 'experiment index')
cmd:option('-reward', 0, 'final reward')
cmd:option('-smooth_val', 0, 'smooth value')
cmd:option('-pretrain_bool', 0, 'pretrained or not')
cmd:option('-pas_num', 10, 'sampled passage number')
cmd:option('-pas_num_pred', 5, 'passage number in prediction')
cmd:option('-reward_epoch', 100, 'get reward')
cmd:option('-train_out', 0, 'output for training data')
cmd:option('-sent_max_len', 500, 'reranking passage max length')
cmd:option('-pt_net', 'spanNet', 'point net')

cmd:option('-wvecDim',300,'embedding dimension')
cmd:option('-mem_dim', 300, 'state dimension')
cmd:option('-att_dim', 300, 'attenion dimension')

cmd:option('-model','rankerReader','model')
cmd:option('-task','quasart','task')

cmd:option('-comp_type', 'submul', 'w-by-w type')

cmd:option('-preEmb','glove','Embedding pretrained method')
cmd:option('-grad','adamax','gradient descent method')

cmd:option('-log', 'nothing', 'log message')

cmd:option('-gpu', 1, 'gpu index')


local opt = cmd:parse(arg)
if opt.gpu ~= -1 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed*1000)
end

torch.manualSeed(opt.seed)
torch.setnumthreads(1)

tr:init(opt)
utils = mprc.Utils()
sample = mprc.Sample(opt)

local vocab = tr:loadVocab(opt.task)
ivocab = tr:loadiVocab(opt.task)

opt.numWords = #ivocab
print ("Vocal size: "..opt.numWords)
print('loading data ..')
local train_dataset = tr:loadData('train', opt.task)
local test_dataset
if opt.task ~= 'squad'  then test_dataset = tr:loadData('test', opt.task) end
local dev_dataset = tr:loadData('dev', opt.task)

torch.manualSeed(opt.seed)

local model_class = mprc[opt.model]

local model = model_class(opt)

if opt.train_out == 1 then  utils:collectTopk(model, {train_dataset, dev_dataset, test_dataset}, opt.task, '../trainedmodel/' .. opt.task .. '1_best_backup') end

local recordTrain, recordTest, recordDev
for i = 1, opt.max_epochs do

    if i == opt.reward_epoch then sample.reward = 1 model.optim_state = { learningRate = 0.001 } model.params:copy( model.best_params ) end

    model:train(train_dataset)
    model.optim_state['learningRate'] = model.optim_state['learningRate'] * opt.lr_decay

    recordDev = model:predict_dataset(dev_dataset)
    model:save('../trainedmodel/', opt, {recordDev}, i)
    if i == opt.max_epochs then
        model.params:copy( model.best_params )
        recordDev = model:predict_dataset(dev_dataset)
        if opt.task ~= 'squad' then
            recordTest = model:predict_dataset(test_dataset, 'test')
        end
        model:save('../trainedmodel/', opt, {recordDev, recordTest}, i)
    end
end
