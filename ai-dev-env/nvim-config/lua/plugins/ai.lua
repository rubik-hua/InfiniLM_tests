return {
    {
        "yetone/avante.nvim",
        build = "make",            -- Builds the Rust backend
        event = "VeryLazy",
        opts = {
            -- Default provider; switch at runtime with :AvanteSwitchProvider
            provider = "claude",
            providers = {
                claude = {
                    endpoint = os.getenv("ANTHROPIC_BASE_URL")
                                or "https://api.anthropic.com",
                    model = os.getenv("ANTHROPIC_MODEL")
                            or "claude-sonnet-4-20250514",
                },
                openai = {
                    endpoint = os.getenv("OPENAI_BASE_URL")
                                or "https://api.openai.com/v1",
                    model = os.getenv("OPENAI_MODEL")
                            or "gpt-4.1",
                },
            },
        },
        build = "make",
        keys = {
            { "<leader>aa", function() require("avante.api").ask() end,
              desc = "Avante: Ask", mode = { "n", "v" } },
            { "<leader>ar", function() require("avante.api").refresh() end,
              desc = "Avante: Refresh" },
            { "<leader>ae", function() require("avante.api").edit() end,
              desc = "Avante: Edit", mode = { "n", "v" } },
        },
        dependencies = {
            "nvim-treesitter/nvim-treesitter",
            "stevearc/dressing.nvim",
            "nvim-lua/plenary.nvim",
            "MunifTanjim/nui.nvim",
            "hrsh7th/nvim-cmp",
            "nvim-tree/nvim-web-devicons",
            -- Copilot support (optional)
            -- "zbirenbaum/copilot.lua",
        },
    },
}
