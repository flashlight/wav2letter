require('nn')
local LogSumExp, parent = torch.class('nn.LogSumExp', 'nn.Module')

function LogSumExp:__init(r)
    parent.__init(self)
    self.iscuda = false
    self.r = r or 1
    self.temp = torch.Tensor()
    self.repeatmax = torch.Tensor()
    self.maxInput = torch.Tensor()
    self.maxId = torch.LongTensor()
    self.sumexp = torch.Tensor()
    self.x = torch.Tensor()
end

function LogSumExp:updateOutput(input)
    if input:dim() == 1 then
        self.output:resize(1)

        local x = input:squeeze()*self.r
        local xptr = x:data()
        local N = input:size(1)

        -- get max values for each class
        local maxid = 0
        local maxInput = xptr[0]
        for i = 1, N-1 do
            if xptr[i] > maxInput then
                maxid = i
                maxInput = xptr[i]
            end
        end
        self.maxInput = maxInput
        self.maxid = maxid

        -- compute LSE
        local sumExp = 0
        for i = 0, N-1 do
            if i ~= self.maxid then
                sumExp = sumExp + math.exp(xptr[i] - self.maxInput)
            end
        end
        self.output:fill( self.maxInput + torch.log1p(sumExp) )
        self.output:add(-math.log(N))
        self.output:div(self.r)

    elseif input:dim() == 2 then
        torch.mul(self.x, input, self.r)
        local x = self.x
        local xptr = x:data()
        local N, nbClass = x:size(1), x:size(2)

        self.output:resize(nbClass)
        local outputptr = self.output:data()

        -- get max values for each class
        local maxid = torch.linspace(0,nbClass-1,nbClass)
        local maxidptr = maxid:data()
        local maxInput = x[{1,{}}]:clone()
        local maxInputptr = maxInput:data()

        self.maxId = self.maxId:cudaLong()

        torch.max(self.maxInput, self.maxId, x, 1)
        self.maxInput = self.maxInput:squeeze()
        self.maxId = self.maxId:squeeze()

        -- compute LSE
        torch.add(self.sumexp, x, torch.repeatTensor(self.repeatmax, self.maxInput, N,1):mul(-1))
        torch.exp(self.temp, self.sumexp)
        torch.sum(self.sumexp, self.temp, 1)
        torch.add(self.output, self.sumexp, -1):log1p():add(self.maxInput)
        self.output:add(-math.log(N))
        self.output:div(self.r)
        self.output:squeeze(self.output)
    elseif input:dim() == 3 then
        if params.iscuda then error('LogSumExp : cuda not implemented for dim>2') end
        local x = input*self.r
        local xptr = x:data()
        local nbClass, M, N = x:size(1), x:size(2), x:size(3)

        self.output:resize(nbClass)
        local outputptr = self.output:data()

        -- get max values for each class
        local maxid = torch.linspace(0,(nbClass-1)*M*N,nbClass)
        local maxidptr = maxid:data()
        local maxInput = x[{{},1,1}]:clone()
        local maxInputptr = maxInput:data()

        for c = 0, nbClass-1 do
            for i = 0, M-1 do
                for j = 0, N-1 do
                    local id = c*M*N + i*N + j
                    if xptr[id] > maxInputptr[c] then
                        maxidptr[c] = id
                        maxInputptr[c] = xptr[id]
                    end
                end
            end
        end
        self.maxInput = maxInput
        self.maxId = maxid

        -- compute the LSE
        for c = 0, nbClass-1 do
            local sumExp = 0
            for i = 0, M-1 do
                for j = 0, N-1 do
                    local id = c*M*N + i*N + j
                    if id ~= maxidptr[c] then
                        sumExp = sumExp + math.exp(xptr[id] - maxInputptr[c])
                    end
                end
            end
            outputptr[c] = maxInputptr[c] + torch.log1p(sumExp)
        end
        self.output:add(-math.log(M*N))
        self.output:div(self.r)

    elseif input:dim() == 4 then
        if params.iscuda then error('LogSumExp : cuda not implemented for dim>2') end

        local x = input*self.r
        local xptr = x:data()
        local batchSize, nbClass, M, N = x:size(1), x:size(2), x:size(3),
            x:size(4)

        self.output:resize(batchSize,nbClass)
        local outputptr = self.output:data()

        -- get max values for each class
        local maxid = torch.linspace(0,(nbClass*batchSize-1)*M*N,nbClass*batchSize)
        local maxidptr = maxid:data()
        local maxInput = x[{{},{},1,1}]:clone()
        local maxInputptr = maxInput:data()
        for b = 0, batchSize-1 do
            for c = 0, nbClass-1 do
                local idPosMax = b*nbClass+c
                for i = 0, M-1 do
                    for j = 0, N-1 do
                        -- print(i,j,c,b)
                        local id = b*nbClass*M*N + c*M*N + i*N + j
                        if xptr[id] > maxInputptr[idPosMax] then
                            maxidptr[idPosMax] = id
                            maxInputptr[idPosMax] = xptr[id]
                        end
                    end
                end
            end
        end
        self.maxInput = maxInput
        self.maxId = maxid

        -- compute the LSE
        for b = 0, batchSize-1 do
            for c = 0, nbClass-1 do
                local sumExp = 0
                local idPosMax = b*nbClass+c
                for i = 0, M-1 do
                    for j = 0, N-1 do
                        local id = b*nbClass*M*N + c*M*N + i*N + j
                        if id ~= maxidptr[idPosMax] then
                            sumExp = sumExp + math.exp(xptr[id]
                                - maxInputptr[idPosMax])
                        end
                    end
                end
                outputptr[idPosMax] = maxInputptr[idPosMax]+torch.log1p(sumExp)
            end
        end
        self.output:add(-math.log(M*N))
        self.output:div(self.r)
    end

    return self.output
end


function LogSumExp:updateGradInput(input,gradOutput)
    self.gradInput:resize(input:size())
    self.gradInput:fill(1)

    local gradInputptr = self.gradInput:data()
    local gradOutputptr = gradOutput:data()

    if input:dim() == 1 then
        local x=input*self.r
        local xptr = x:data()
        local N = x:size(1)
        local sumExp = 0
        for i = 0,N-1 do
            if i ~= self.maxId then
                local z = math.exp(xptr[i]-self.maxInput)
                sumExp = sumExp + z
                gradInputptr[i] = z
            end
        end
        self.gradInput:div(1+sumExp)
        self.gradInput:mul(gradOutput[1])

    elseif input:dim() == 2 then
        local x = self.x
        local N,nbClass = x:size(1), x:size(2)

        local xptr = x:data()
        local maxidptr = self.maxId:data()
        local maxInputptr = self.maxInput:data()

        if false then -- susceptible to explode...
            self.gradInput:copy(x)
            self.gradInput:add(self.repeatmax):exp()--repeatmax should already contain self.maxInput:repeatTensor(N,1))
            self.gradInput:cdiv(self.temp:repeatTensor(self.sumexp,N,1))
            self.gradInput:cmul( torch.repeatTensor(self.temp, gradOutput, N,1))--:mul(-1) --wtf : error lower if I add it...
            --print(self.gradInput)
            --io.read()
        else -- equivalent but explosion proof
            self.gradInput:fill(1)
            --print(self.gradInput)
            --print(input)
            --print(self.repeatmax)
            self.repeatmax:mul(-1) -- note: to be more efficient, store self.repeatmax before the mul(-1)
            --print(self.repeatmax)
            --print(self.temp)
            --print(self.sumexp:size())
            --print(self.sumexp)
            self.temp:copy(x):mul(-1)
            --print(tutu)
            -- print('toto')
            self.temp:add(self.repeatmax)
            --print(self.temp)
            self.temp:exp()
            --print(self.temp)
            torch.repeatTensor(self.repeatmax, self.sumexp, N,1) --we don't need self.repeatmax anymore. I use it as a temp.
            self.temp:cmul(self.repeatmax)
            --print(self.temp)
            self.gradInput:cdiv(self.temp)
            --print(self.gradInput)
            self.gradInput:cmul( torch.repeatTensor(self.temp, gradOutput, N,1))
            --print(self.gradInput)
            self.gradInput:squeeze(self.gradInput)
        end
    elseif input:dim() == 3 then
        local x = input*self.r
        local nbClass, M, N = x:size(1), x:size(2), x:size(3)

        local xptr = x:data()
        local maxidptr = self.maxId:data()
        local maxInputptr = self.maxInput:data()

        for c=0,nbClass-1 do
            local sumExp = 0
            for i = 0, M-1 do
                for j = 0, N-1 do
                    local id = c*M*N + i*N + j
                    if id ~= maxidptr[c] then
                        local z = math.exp(xptr[id]-maxInputptr[c])
                        sumExp = sumExp + z
                        gradInputptr[id] = z
                    end
                end
            end
            self.gradInput:select(1,c+1):mul(gradOutputptr[c]/(1+sumExp))
        end

    elseif input:dim() == 4 then
        local x = input*self.r
        local batchSize, nbClass, M, N = x:size(1),x:size(2),x:size(3),x:size(4)

        local xptr = x:data()
        local maxidptr = self.maxId:data()
        local maxInputptr = self.maxInput:data()

        for b = 0, batchSize-1 do
            for c=0,nbClass-1 do
                local sumExp = 0
                local idPosMax = b*nbClass+c
                for i = 0, M-1 do
                    for j = 0, N-1 do
                        local id = b*nbClass*M*N + c*M*N + i*N + j
                        if id ~= maxidptr[idPosMax] then
                            local z = math.exp(xptr[id]-maxInputptr[idPosMax])
                            sumExp = sumExp + z
                            gradInputptr[id] = z
                        end
                    end
                end
                self.gradInput:select(2,c+1):mul(gradOutputptr[idPosMax]/(1+sumExp))
            end
        end
    end
    return self.gradInput
end

function LogSumExp:cuda(input,gradOutput)
    print('caution : experimental and only for input dim==1 or 2')
    self.output = self.output:cuda()
    self.gradInput = self.gradInput:cuda()
    self.temp = self.temp:cuda()
    self.repeatmax = self.repeatmax:cuda()
    self.maxInput = self.maxInput:cuda()
    self.maxId = self.maxId:cuda()
    self.sumexp = self.sumexp:cuda()
    self.x = self.x:cuda()
    self.iscuda = true
    return self
end
