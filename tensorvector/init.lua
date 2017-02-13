--[[

This is a simple vector implementation on top of Torch. It is serializable so
you can save it in a file, and it can be stored in tds.hash.

]]--


-- dependencies:
require 'torch'
local ffi = require 'ffi'
local _tds = require 'tds'
local elem = require 'tds.elem'

-- function that defines TensorVector for a specific type:
local function defineTensorVector(typename)

    -- define C type:
    local storagename = string.format('%sStorage',             typename)
    local tensorname  = string.format('%sTensor',              typename)
    local vectorname  = string.format('%sVector',              typename)
    local cvectorname = string.format('TH%sVector',            typename)
    local storagefree = string.format('TH%sStorage_free',      typename)
    local setflag     = string.format('TH%sStorage_setFlag',   typename)
    local clearflag   = string.format('TH%sStorage_clearFlag', typename)
    local cdef = [[
        typedef struct {
          THRealStorage *storage;
          THLongStorage *size;
        } THRealVector;

        void THRealStorage_free(THRealStorage *storage);
        void THLongStorage_free(THLongStorage *storage);

        void THRealStorage_setFlag(THRealStorage *storage, const char flag);
        void THRealStorage_clearFlag(THRealStorage *storage, const char flag);

        void *malloc(size_t size);
        void free(void *ptr);
    ]]
    cdef = cdef:gsub('Real', typename)
    ffi.cdef(cdef)

    -- register C-structure in tds:
    local mt = {}
    local function free_p(celem)    -- frees the C-structure
        celem = ffi.cast(cvectorname .. '*', celem)
        ffi.C[storagefree](celem.storage)
        ffi.C['THLongStorage_free'](celem.size)
        ffi.C.free(celem)
    end
    free_p = ffi.cast('void (*)(void*)', free_p)
    local function set_func(lelem)  -- sets the C-structure
        local celem = ffi.cast(
            cvectorname .. '*',
            ffi.C.malloc(ffi.sizeof(cvectorname))
        )
        celem.storage = lelem.storage:cdata()
        lelem.storage:retain()
        celem.size = lelem.size:cdata()
        lelem.size:retain()
        return celem, free_p
    end
    local function get_func(celem)   -- gets the C-structure
        local celem = ffi.cast(cvectorname .. '*', celem)
        local lelem = {}
        lelem.size    = torch.pushudata(celem.size, 'torch.LongStorage')
        lelem.size:retain()
        lelem.storage = torch.pushudata(celem.storage, 'torch.' .. storagename)
        lelem.storage:retain()
        setmetatable(lelem, mt)
        return lelem
    end
    elem.addctype(  -- adds type as a tds element
        vectorname,
        free_p,
        set_func,
        get_func
    )

    -- define all functions in the metatable:
    function mt.__new(initialCapacity)
        local self = {}
        local initialCapacity = initialCapacity or 1
        self.storage = torch[storagename](initialCapacity)
        self.size = torch.LongStorage(1)
        self.size[1] = 0   -- using a number does not play nice with tds.hash
        self.STORAGE_RESIZABLE = 2
        setmetatable(self, mt)
        return self
    end

    function mt:__index(k)
        assert(self)
        if type(k) == 'string' then return rawget(mt, k) end
        if k <= 0 or k > self.size[1] then error('index out of bounds') end
        return self.storage[k]
    end

    function mt:__newindex(k, v)
        assert(self)
        assert(type(k) == 'number')
        if k <= 0 then error('index out of bounds') end
        if not v then error('removal not supported') end
        if k >= self.storage:size() then
            self.storage:resize((self.storage:size() + 1) * 2)
        end
        self.storage[k] = v
        self.size[1] = math.max(self.size[1], k)
    end

    function mt:__len()
        assert(self)
        return self.size[1]
    end

    function mt:__write(f)
        assert(self)
        self:compress()
        f:writeLong(self.size[1])
        f:writeObject(self.storage)
    end

    function mt:__read(f)
        assert(self)
        self.size[1] = f:readLong()
        self.storage:resize(self.size[1]):copy(f:readObject())
    end

    function mt:__tostring()
        assert(self)
        return torch.typename(self)
    end

    function mt:__pairs()
        assert(self)
        local k = 0
        return function()
            k = k + 1
            if k <= self.size[1] then return k, self.storage[k] end
        end
    end

    mt.__version = 0
    mt.__typename = vectorname
    mt.__factory = function(file) return mt.__new() end

    local function __insertTensor(self, k, tensor)
        assert(self)
        assert(type(k) == 'number')
        assert(torch.typename(tensor) == 'torch.' .. tensorname)
        local tensor = tensor:resize(tensor:nElement())
        while k + tensor:nElement() - 1 > self.storage:size() do
            self.storage:resize((self.storage:size() + 1) * 2)
        end
        torch[tensorname](self.storage):narrow(
            1, k, tensor:nElement()
        ):copy(tensor)
        self.size[1] = math.max(self.size[1], k + tensor:nElement() - 1)
    end

    function mt:insertTensor(k, tensor)
        assert(self)
        if k and tensor then __insertTensor(self, k, tensor)
        elseif k then __insertTensor(self, #self + 1, k)
        else error('Incorrect inputs.') end
    end

    function mt:getStorage()
        assert(self)
        self:compress()
        return self.storage
    end

    function mt:getTensor()
        assert(self)
        self:compress()
        return torch[tensorname](self.storage)
    end

    function mt:compress()
        assert(self)
        self:resize(self.size[1])
    end

    function mt:resize(newSize)
        assert(self)
        assert(newSize >= 0)
        ffi.C[setflag](self.storage:cdata(), self.STORAGE_RESIZABLE or 2)
        self.storage:resize(newSize)
        self.size[1] = math.min(self.size[1], newSize)
    end

    -- copy all Tensor functions:
    for funcname, func in pairs(torch.getmetatable('torch.' .. tensorname)) do
        if funcname:sub(1, 1) ~= '_'
        and funcname ~= 'resize'
        and funcname ~= 'resizeAs'
        and funcname ~= 'reshape' then
            mt[funcname] = function(self, ...)
                assert(self)
                ffi.C[clearflag](
                   self.storage:cdata(), self.STORAGE_RESIZABLE or 2
                )
                local t = self:getTensor()
                local result = t[funcname](t, ...)
                ffi.C[setflag](
                   self.storage:cdata(), self.STORAGE_RESIZABLE or 2
                )
                return result
            end
        end
    end

    -- register type in Torch:
    torch.metatype(vectorname, mt)
    return mt.__new
end

-- generate constructors for all Tensor types:
local M = {}
local typenames = {'Char', 'Byte', 'Short', 'Int', 'Long', 'Float', 'Double'}
for _,val in pairs(typenames) do
    M['new_' .. val:lower()] = defineTensorVector(val)
end

-- return module:
return M
