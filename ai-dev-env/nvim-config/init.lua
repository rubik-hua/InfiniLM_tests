-- ============================================================================
-- General Settings
-- ============================================================================
vim.opt.relativenumber = false   -- Absolute line numbers keep things grounded
vim.opt.number = true
vim.opt.cursorline = true        -- Highlight the current line
vim.opt.termguicolors = true     -- True color support
vim.opt.signcolumn = "yes"       -- Prevent the sign column from shifting
vim.opt.updatetime = 250         -- Quicker diagnostics and swaps

-- ============================================================================
-- Bootstrap lazy.nvim
-- ============================================================================
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({
        "git",
        "clone",
        "--filter=blob:none",
        "https://github.com/folke/lazy.nvim.git",
        "--branch=stable",
        lazypath,
    })
end
vim.opt.rtp:prepend(lazypath)

-- ============================================================================
-- Leader Keys (Must be set before loading plugins)
-- ============================================================================
vim.g.mapleader = " "
vim.g.maplocalleader = "\\"

-- ============================================================================
-- Plugin Management
-- ============================================================================
require("lazy").setup({
    spec = {
        { "folke/tokyonight.nvim", lazy = false, priority = 1000 },

        -- Load plenary eagerly to prevent "module not found" race conditions
        { "nvim-lua/plenary.nvim", lazy = false },

        -- AI Copilot
        {
            "yetone/avante.nvim",
            lazy = false,          -- Load eagerly to avoid startup sequencing issues
            build = "make",
            opts = {
		provider = "openai",

		-- 2. Detailed provider configurations
		providers = {
		    openai = {
		        endpoint = "http://i-nhi.zhejianglab.org/maas/v1" ,
			model = "GLM-5.1-w8a8",
			timeout = 60000,
		    },
		    -- ========== Zhipu BigModel (OpenAI-compatible API) ==========
		    bigmodel = {
		        -- Inherit from the OpenAI provider and use the OpenAI Chat Completions protocol
		        __inherited_from = "openai",
		
		        -- Zhipu OpenAI-compatible endpoint, up to /v1 (do NOT include /chat/completions)
		        -- Docs: https://open.bigmodel.cn/api/paas/v4/chat/completions
		        endpoint = "https://open.bigmodel.cn/api/paas/v4",
		
		        -- Model name shown in the BigModel console (e.g. glm-4-flash, glm-4.7-flash, glm-5)
		        model = "glm-4-flash",
		
		        -- Name of the environment variable for the API key (using AVANTE_ prefix for scoping)
		        api_key_name = "AVANTE_BIGMODEL_API_KEY",
		
		        -- Extra request parameters (adjust to taste)
		        extra_request_body = {
		            temperature = 0.7,
		            max_tokens = 2048,
		        },
		
		        -- Timeout in ms; increase if the model is slow
		        timeout = 60000,
		    },
		
		    -- ========== Locally deployed LLM (OpenAI-compatible API) ==========
		    local_llm = {
		        __inherited_from = "openai",
		
		        -- Local service base URL, up to /v1, e.g.:
		        --   http://127.0.0.1:8000/v1
		        --   http://192.168.1.100:11434/v1   (Ollama OpenAI-compatible layer)
		        endpoint = "http://127.0.0.1:8000/v1",
		
		        -- Local model name, depends on your deployment (e.g. qwen2.5-coder:7b, deepseek-coder)
		        model = "qwen2.5-coder:7b",
		
		        -- If your local server doesn’t require an API key, just use any env var name here
		        api_key_name = "AVANTE_LOCAL_LLM_API_KEY",
		
		        extra_request_body = {
		            temperature = 0.6,
		            max_tokens = 4096,
		        },
		
		        timeout = 120000,  -- Local models may be slower, give more time
		    }

	    	},
    	    },
            dependencies = {
                "stevearc/dressing.nvim",
	        "MunifTanjim/nui.nvim",
	        {
	    	    "MeanderingProgrammer/render-markdown.nvim",
	    	    opts = { file_types = { "markdown", "Avante" } },
	        },
	   },
        },
    },
    defaults = { lazy = false },
    install = { colorscheme = { "habamax" } },
})

-- ============================================================================
-- Colorscheme
-- ============================================================================
vim.cmd([[colorscheme tokyonight-night]])

