
local stringx = require 'pl.stringx'
require 'debug'
require 'paths'

tr = {}
tr.__index = tr
function tr:init(opt)
    self.task_list = opt.task_list
    if not paths.filep("../data/".. opt.task .."/vocab.t7") then
        self:buildVocab(opt.task)
    end

    if not paths.filep("../data/".. opt.task .."/initEmb.t7") then
        self:buildVacab2Emb(opt)
    end

    if not paths.filep("../data/".. opt.task .."/sequence/dev.t7") then
        tr:buildData('all', opt.task)
    end

end

function tr:loadVocab(task)
    return  torch.load("../data/".. task .."/vocab.t7")
end

function tr:loadUnUpdateVocab(task)
    return  torch.load("../data/".. task .."/unUpdateVocab.t7")
end

function tr:loadiVocab(task)
    return torch.load("../data/"..task.."/ivocab.t7")
end

function tr:loadVacab2Emb(task)
    print("Loading embedding ...")
    return torch.load("../data/"..task.."/initEmb.t7")
end

function tr:loadEntVocab(task)
    return  torch.load("../data/".. task .."/entvocab.t7")
end

function tr:loadiEntVocab(task)
    return torch.load("../data/"..task.."/ientvocab.t7")
end

function tr:loadData(filename, task)
    print("Loading data "..filename.."...")
    local data
    local tasks = {quasart=1,squad=1,quasartans=1,searchqa=1,searchqaans=1,squadans=1,triviaqa=1,triviaqaans=1,triviaqaweb=1,triviaqaweb2=1,triviaqawebans=1,unftriviaqa=1,unftriviaqaans=1}
    if tasks[task] ~= nil then
        data = torch.load("../data/"..task.."/sequence/"..filename..".t7")
    else
        error('The specified task is not supported yet!')
    end
    return data
end

function tr:buildVocab(task)
    print ("Building vocab dict ...")
    if task == 'quasart' or task == 'searchqa' or task == 'triviaqa' or task == 'triviaqaweb' or task == 'unftriviaqa' then
        local tokens = {}
        local index = {}
        tokens['<<UNK>>'] = 2
        tokens['<<PADDING>>'] = 1
        index[2] = '<<UNK>>'
        index[1] = '<<PADDING>>'
        local count = 1
        local filenames = {"../data/"..task.."/sequence/train.tsv", "../data/"..task.."/sequence/dev.tsv", "../data/"..task.."/sequence/test.tsv"}

        for _, filename in pairs(filenames) do
            for line in io.lines(filename) do
                if line ~= '*new_instance*' then
                    local sents = stringx.split(line, '*split_sign*')
                    local words = stringx.split(sents[1], ' ')
                    for _, word in pairs(words) do
                        if tokens[word] == nil then
                            tokens[word] = count
                            index[count] = word

                            count = count + 1
                        end
                    end
                end
            end
        end
        print(#index)
        torch.save("../data/"..task.."/vocab.t7", tokens)
        torch.save("../data/"..task.."/ivocab.t7", index)
        torch.save("../data/"..task.."/vocab_all.t7", tokens)
        torch.save("../data/"..task.."/ivocab_all.t7", index)

        local file = io.open("../data/"..task.."/vocab.txt","w")
        for word, i in pairs(tokens) do
            file:write(word .. ' ' .. i .. '\n')
        end
        file:close()
    elseif task == 'squad' then
        local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}
        local vocab = {}
        local ivocab = {}
        local a_vocab = {}
        local a_ivocab = {}
        vocab['<<UNK>>'] = 2
        vocab['<<PADDING>>'] = 1
        ivocab[2] = '<<UNK>>'
        ivocab[1] = '<<PADDING>>'
        for _, filename in pairs(filenames) do
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                for j = 1, 2 do
                    local words = stringx.split(divs[j], ' ')
                    for i = 1, #words do
                        if vocab[words[i]] == nil then
                            vocab[words[i]] = #ivocab + 1
                            ivocab[#ivocab + 1] = words[i]
                        end
                    end
                end
            end
        end
        torch.save("../data/"..task.."/vocab.t7", vocab)
        torch.save("../data/"..task.."/ivocab.t7", ivocab)

    else
        error('The specified task is not supported yet!')
    end
end

function tr:buildVacab2Emb(opt)
    if opt.task == 'quasart' or opt.task == 'searchqa' or opt.task == 'triviaqa' or opt.task == 'triviaqaweb' or opt.task == 'unftriviaqa'  then
        local vocab_all = self:loadVocab(opt.task)
        local ivocab_all = self:loadiVocab(opt.task)
        local emb = torch.zeros(#ivocab_all, 300)
        print ("Loading ".. opt.preEmb .. " ...")
        local file
        if opt.preEmb == 'glove' then
            file = io.open("../data/"..opt.preEmb.."/glove.840B.300d.txt", 'r')
        end

        local count = 0
        local vocab = {}
        local ivocab = {}
        vocab['<<UNK>>'] = 2
        vocab['<<PADDING>>'] = 1
        ivocab[2] = '<<UNK>>'
        ivocab[1] = '<<PADDING>>'

        while true do
            local line = file:read()

            if line == nil then break end
            vals = stringx.split(line, ' ')
            if vocab_all[vals[1]] ~= nil then
                for i = 2, #vals do
                    emb[#ivocab + 1][i-1] = tonumber(vals[i])
                end

                vocab[vals[1]] = #ivocab + 1
                ivocab[#ivocab + 1] = vals[1]

                count = count + 1
                if count == #ivocab_all then
                    break
                end
            end
        end
        emb = emb:sub(1, #ivocab)
        print("Number of words not appear in glove: "..(#ivocab_all-count) )
        if vocab['<<PADDING>>'] ~= nil then emb[vocab['<<PADDING>>']]:zero() end
        if vocab['<<UNK>>'] ~= nil then emb[vocab['<<UNK>>']]:zero() end
        torch.save("../data/"..opt.task.."/initEmb.t7", emb)
        torch.save("../data/"..opt.task.."/vocab.t7", vocab)
        torch.save("../data/"..opt.task.."/ivocab.t7", ivocab)
    else
        local vocab = self:loadVocab(opt.task)
        local ivocab = self:loadiVocab(opt.task)
        local emb = torch.randn(#ivocab, opt.wvecDim) * 0.05

        print ("Loading ".. opt.preEmb .. " ...")
        local file
        if opt.preEmb == 'glove' then
            file = io.open("../data/"..opt.preEmb.."/glove.840B.300d.txt", 'r')
        end

        local count = 0
        local embRec = {}
        while true do
            local line = file:read()

            if line == nil then break end
            vals = stringx.split(line, ' ')
            if vocab[vals[1]] ~= nil then
                for i = 2, #vals do
                    emb[vocab[vals[1]]][i-1] = tonumber(vals[i])
                end
                embRec[vocab[vals[1]]] = 1
                count = count + 1
                if count == #ivocab then
                    break
                end
            end
        end
        print("Number of words not appear in ".. opt.preEmb .. ": "..(#ivocab-count) )

        torch.save("../data/"..opt.task.."/initEmb.t7", emb)
        torch.save("../data/"..opt.task.."/unUpdateVocab.t7", embRec)
    end
end


function tr:buildData(filename, task)
    local trees = {}
    local lines = {}
    local dataset = {}
    idx = 1

    print ("Building "..task.." "..filename.." data ...")

    if task == 'quasart' or task == 'searchqa' or task == 'unftriviaqa'  then
        local vocab = torch.load('../data/' .. task .. '/vocab.t7')
        local ivocab = torch.load('../data/' .. task .. '/ivocab.t7')
        local vocab_all = torch.load('../data/' .. task .. '/vocab_all.t7')
        local ivocab_all = torch.load('../data/' .. task .. '/ivocab_all.t7')

        local filenames = {dev='../data/' .. task .. '/sequence/dev.tsv', test = '../data/' .. task .. '/sequence/test.tsv', train='../data/' .. task .. '/sequence/train.tsv'}
        --local filenames = {dev = '../data/' .. task .. '/sequence/dev.tsv', train='../data/' .. task .. '/sequence/train.tsv'}

        local large_num = 0
        local passage_num = 0
        local passage_max_sent_len = 50

        local question_perfile
        if task == 'quasart' then question_perfile = 14500 elseif task == 'unftriviaqa'  then question_perfile = 30000 else question_perfile = 50000 end

        local file_train_testing = io.open('../data/' .. task .. '/sequence/train4test1_testing.txt', 'w')

        for set_name, filename in pairs(filenames) do
            local dev_acc = 0.0
            local total_num = 0.0
            local data = {}

            local train4test = {}
            local train_testing = {}


            local data_idx = 1
            local instance = {}
            for i = 1, 5 do instance[i] = {} end
            local question_tensor, passages_pos, passages_pos_idx, passages_neg, passages_neg_idx, passages_pts = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
            local passages_pn, passages_pn_idx, passages_pn_all = torch.Tensor(), torch.Tensor(), torch.Tensor()

            local words_label, labelall

            local question, question_id
            local true_passages = {}
            local question_bool = true
            for line in io.lines(filename) do
                if line == '*new_instance*' or line:sub(-14,-1) == '*new_instance*' then
                --if line == '' then
                    if passages_pts:dim() > 0 and set_name == 'train' then
                        data[#data+1] = {question_tensor, passages_pos, passages_pos_idx, passages_neg, passages_neg_idx, passages_pts}
                        assert(passages_pos_idx:size(1) == passages_pts:size(1))

                    elseif set_name == 'dev' or set_name == 'test' then
                        data[#data+1] = {question_tensor, passages_pn, passages_pn_idx, passages_pn_all, question_id}
                    end

                    if set_name == 'train' then train4test[#train4test+1] = {question_tensor, passages_pn, passages_pn_idx, passages_pn_all, question_id} end

                    if passages_pts:dim() ~= 0 then dev_acc = dev_acc + 1 end
                    total_num = total_num + 1
                    question_tensor, passages_pos, passages_pos_idx, passages_neg, passages_neg_idx, passages_pts = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
                    passages_pn, passages_pn_idx, passages_pn_all = torch.Tensor(), torch.Tensor(), torch.Tensor()
                    question_bool = true
                else
                    local divs = stringx.split(line, '*split_sign*')
                    --local divs = stringx.split(line, '\t')
                    if #divs == 0 then print(line) end
                    if #divs ~= 2 and #divs ~= 3 then
                        print(divs)
                        print(#divs)
                        print(question_bool)
                    end
                    --assert(#divs == 2 or #divs == 3)
                    if #divs == 2 then
                        assert(question_bool == false)
                        local words = stringx.split(divs[1], ' ')
                        if #words ~= 0 then
                            --furtherSplit(words)
                            local pts = self:matchAnswer(words, words_label)

                            local ls = stringx.split(labelall, "*split_answer*")
                            --local ls = stringx.split(labelall, "\\E|\\Q")

                            if #ls ~= 1 and pts:dim()==0 then
                                for k = 2, #ls do
                                    pts = self:matchAnswer(words, stringx.split(ls[k], ' '))
                                    if pts:dim()~=0 then break end
                                end
                            end

                            local sent_tensor = torch.LongTensor(#words)
                            local sent_all_tensor = torch.LongTensor(#words)
                            for j, word in pairs(words) do
                                if vocab[word] then sent_tensor[j] = vocab[word] else sent_tensor[j] = 2 end
                                sent_all_tensor[j] = vocab_all[word]--or 2
                            end
                            if pts:dim()~= 0 then passages_pts = passages_pts:dim()~=0 and torch.cat(passages_pts, pts) or pts end
                            if set_name == 'train' then
                                if pts:dim() ~= 0 then
                                    local idx = passages_pos:dim() ~= 0 and torch.LongTensor{passages_pos:size(1)+1, passages_pos:size(1)+sent_tensor:size(1)} or torch.LongTensor{1, sent_tensor:size(1)}
                                    passages_pos_idx = passages_pos_idx:dim() ~= 0 and torch.cat(passages_pos_idx, idx) or idx
                                    passages_pos = passages_pos:dim() ~= 0 and torch.cat(passages_pos, sent_tensor) or sent_tensor
                                else
                                    local idx = passages_neg:dim() ~= 0 and torch.LongTensor{passages_neg:size(1)+1, passages_neg:size(1)+sent_tensor:size(1)} or torch.LongTensor{1, sent_tensor:size(1)}
                                    passages_neg_idx = passages_neg_idx:dim() ~= 0 and torch.cat(passages_neg_idx, idx) or idx
                                    passages_neg = passages_neg:dim() ~= 0 and torch.cat(passages_neg, sent_tensor) or sent_tensor
                                end
                            end
                            local idx = passages_pn_idx:dim() ~= 0 and torch.LongTensor{passages_pn:size(1)+1, passages_pn:size(1)+sent_tensor:size(1)} or torch.LongTensor{1, sent_tensor:size(1)}
                            passages_pn_idx = passages_pn_idx:dim() ~= 0 and torch.cat(passages_pn_idx, idx) or idx
                            passages_pn = passages_pn:dim() ~= 0 and torch.cat(passages_pn, sent_tensor) or sent_tensor
                            passages_pn_all = passages_pn_all:dim() ~= 0 and torch.cat(passages_pn_all, sent_all_tensor) or sent_all_tensor
                            assert(passages_pn:size(1) == passages_pn_all:size(1))
                        --else
                        --    print(line)
                        end
                    elseif question_bool then
                        assert( #divs == 3 )
                        question = divs[1]
                        question_id = divs[2]
                        labelall = divs[3]
                        if set_name ~= 'train' then
                            divs[3] = divs[3] ~= '' and stringx.split(divs[3], "*split_answer*")[1] or divs[3]
                        end
                        if set_name == 'train' then file_train_testing:write(question_id .. '\t' .. labelall .. '\n') end

                        local words_q = stringx.split(divs[1], ' ')

                        if divs[3]:sub(-1,-1) == ',' then divs[3] = divs[3]:sub(1,-2) end
                        words_label = stringx.split(divs[3], ' ')

                        --if #words_label == 0 then print(divs[4]) debug.debug() end

                        if words_label[#words_label] == '' then table.remove(words_label, #words_label) end
                        if words_label[#words_label] == '.' then table.remove(words_label, #words_label) end
                        if words_label[#words_label] == ',' then table.remove(words_label, #words_label) end
                        if words_label[1] == '' then table.remove(words_label, 1) end

                        --furtherSplit(words_label)

                        question_tensor = torch.LongTensor(#words_q)
                        for j, word in pairs(words_q) do
                            if vocab[word] then question_tensor[j] = vocab[word] else question_tensor[j] = 2 end
                        end
                        question_bool = false
                    end
                end

                if #data == question_perfile and set_name == 'train' then
                    torch.save('../data/' .. task .. '/sequence/'..set_name..data_idx..'.t7', data)
                    torch.save('../data/' .. task .. '/sequence/train4test'..data_idx..'.t7', train4test)
                    data_idx = data_idx + 1
                    file_train_testing:close()
                    file_train_testing = io.open('../data/' .. task .. '/sequence/train4test'.. data_idx ..'_testing.txt', 'w')

                    data = {}
                    train4test = {}
                end

            end
            if set_name == 'train' then
                torch.save('../data/' .. task .. '/sequence/'..set_name..data_idx..'.t7', data)
                torch.save('../data/' .. task .. '/sequence/train4test'..data_idx..'.t7', train4test)
                torch.save('../data/' .. task .. '/sequence/'..set_name..'.t7', {data_idx, question_perfile*(data_idx-1) + #data})
                file_train_testing:close()
            else
                torch.save('../data/' .. task .. '/sequence/'..set_name..'.t7', data)
            end

            print(set_name)
            print(dev_acc)
            print(total_num)
            print("Percentage of questions containing answer in the retrieved passages: " .. (dev_acc * 100 / total_num) .. "%")
        end
    elseif task == 'quasartans' or task == 'searchqaans'  or task == 'unftriviaqaans' then
        local vocab = torch.load('../data/'..task..'/vocab.t7')
        local ivocab = torch.load('../data/'..task..'/ivocab.t7')
        local vocab_all = torch.load('../data/'..task..'/vocab_all.t7')
        local ivocab_all = torch.load('../data/'..task..'/ivocab_all.t7')

        local question_perfile = 30000

        local data_idx = 1

        local filenames = {dev='../data/'..task..'/sequence/dev.tsv', test = '../data/'..task..'/sequence/test.tsv', train='../data/'..task..'/sequence/train.tsv'}

        if task == 'squadans' then
            filenames = {dev='../data/'..task..'/sequence/dev.tsv', train='../data/'..task..'/sequence/train.tsv'}
        end

        local question, question_id, labelall, question_tensor, true_id
        local passages_pn_idx, passages_pn = torch.LongTensor(), torch.LongTensor()
        local answers = {}
        local true_ids = {}
        local passage_id = 1
        for set_name, filename in pairs(filenames) do
            print(set_name)
            local data = {}
            local bool_question = true
            for line in io.lines(filename) do

                if line == '*new_instance*' then

                    if set_name == 'train' then
                        if passages_pn_idx:dim()~=0 and question_tensor:dim() ~= 0 then
                            --assert(true_id ~= nil)
                            if true_id ~= nil then
                                data[#data + 1] = {question_tensor, passages_pn_idx, passages_pn, answers, question_id, true_id, true_ids}
                            else
                                print(labelall)
                            end
                            true_id = nil
                        else
                            print(question_tensor)
                        end
                        true_ids = {}
                    else
                        if question_tensor:dim() ~= 0 then
                            assert(passages_pn_idx:dim()~=0)
                            data[#data + 1] = {question_tensor, passages_pn_idx, passages_pn, answers, question_id }
                        else
                            print(question_tensor)
                        end
                    end

                    if #data == question_perfile and set_name == 'train' then
                        torch.save('../data/' .. task .. '/sequence/'..set_name..data_idx..'.t7', data)
                        data_idx = data_idx + 1
                        data = {}
                    end

                    passages_pn_idx, passages_pn = torch.LongTensor(), torch.LongTensor()
                    answers = {}
                    bool_question = true
                else
                    local divs = stringx.split(line, '*split_sign*')
                    if #divs == 0 then print(line) end
                    if bool_question then
                        assert(#divs == 3)
                        question = divs[1]
                        question_id = divs[2]
                        labelall = divs[3] ~= '' and stringx.split(divs[3], "*split_answer*") or divs[3]

                        local words_q = stringx.split(divs[1], ' ')

                        question_tensor = torch.LongTensor(#words_q)

                        for j, word in pairs(words_q) do
                            if vocab[word] then question_tensor[j] = vocab[word] else question_tensor[j] = 2 end
                        end
                        bool_question = false
                    else
                        local ans_words = stringx.split(divs[1], ' ')
                        if #ans_words ~= 0 then
                            for k, v in pairs(labelall) do
                                if v:lower() == divs[1]:lower() then true_id = #answers + 1  true_ids[#true_ids+1] = #answers + 1 break end
                            end

                            passages = torch.LongTensor()
                            for p = 3, #divs do
                                local words = stringx.split(divs[p], ' ')
                                local sent_tensor = torch.LongTensor(#words)
                                for j, word in pairs(words) do
                                    if vocab[word] then sent_tensor[j] = vocab[word] else sent_tensor[j] = 2 end
                                end
                                passages = passages:dim() ~= 0 and torch.cat(passages, sent_tensor) or sent_tensor
                            end

                            local idx = passages_pn_idx:dim() ~= 0 and torch.LongTensor{passages_pn:size(1)+1, passages_pn:size(1)+passages:size(1)} or torch.LongTensor{1, passages:size(1)}
                            passages_pn_idx = passages_pn_idx:dim() ~= 0 and torch.cat(passages_pn_idx, idx) or idx
                            passages_pn = passages_pn:dim() ~= 0 and torch.cat(passages_pn, passages) or passages


                            local ans_tensor = torch.LongTensor(#ans_words)
                            for k = 1, #ans_words do ans_tensor[k] = vocab[ans_words[k]] or 2 end
                            answers[#answers + 1] = {divs[1], ans_tensor, divs[2]}
                        --else
                        --    print(set_name..divs[1]..set_name)
                        end
                    end
                end
            end
            if set_name == 'train' then
                torch.save('../data/' .. task .. '/sequence/'..set_name..data_idx..'.t7', data)
                torch.save('../data/' .. task .. '/sequence/'..set_name..'.t7', {data_idx, question_perfile*(data_idx-1) + #data})
            else
                torch.save('../data/' .. task .. '/sequence/'..set_name..'.t7', data)
            end

        end
    elseif task == 'squad' then
        local vocab = tr:loadVocab(task)
        --local entvocab = self:loadEntVocab(task)
        local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}
        for folder, filename in pairs(filenames) do
            local data = {}
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                local instance = {}
                for j = 1, 2 do
                    local words = stringx.split(divs[j], ' ')
                    instance[j] = torch.IntTensor(#words)
                    for i = 1, #words do
                        instance[j][i] = vocab[ words[i] ]
                    end
                end
                if folder == 'train' then
                    local pos = stringx.split(stringx.strip(divs[3]), ' ')
                    instance[3] = torch.IntTensor(#pos+1)
                    for i = 1, #pos do
                        instance[3][i] = tonumber(pos[i])
                    end
                    instance[3][#pos+1] = instance[1]:size(1)+1
                else
                    instance[3] = 'null'
                end
                --[[
                for j = 6, 7 do
                    local words = stringx.split(divs[j], ' ')
                    instance[j-2] = torch.IntTensor(#words)
                    for i = 1, #words do
                        instance[j-2][i] = entvocab[ words[i] ]
                    end
                end
                ]]
                data[#data+1] = instance
            end
            torch.save("../data/"..task.."/sequence/"..folder..'.t7', data)
        end
    else
        error('The specified task is not supported yet!')
    end
    return dataset
end

function tr:matchAnswer(words, answer)
    local answer_string = table.concat(answer, ' '):lower()
    local answer_len = #answer
    if answer_len > 0 then
        for i = 1, #words - answer_len + 1 do
            cand_string = table.concat(words, ' ', i, i+answer_len-1):lower()
            if cand_string == answer_string then
                return torch.LongTensor{i, i+answer_len-1}
            end
        end
    end

    return torch.LongTensor()
end

return tr
