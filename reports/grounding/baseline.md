# Target GUI grounding baseline

- created_at: `2026-08-13T09:50:23.209078+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **none** (no RAG / no LoRA)
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/grounding.json`
- samples: 32

| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |
|---|---:|---:|---:|---:|---:|
| overall | 32 | 75.0% | 42325 | 1322.7 | 0 |

## By screen

| screen | n | accuracy | avg_tokens |
|---|---:|---:|---:|
| account_logged_in | 3 | 66.7% | 1320.7 |
| cart_with_item | 3 | 100.0% | 1322.3 |
| category | 1 | 100.0% | 1323.0 |
| contact | 2 | 0.0% | 1323.0 |
| forgot_password | 1 | 100.0% | 1320.0 |
| home | 9 | 77.8% | 1323.3 |
| login | 3 | 100.0% | 1322.7 |
| minicart_open | 2 | 100.0% | 1327.0 |
| orders_returns | 1 | 100.0% | 1320.0 |
| product | 4 | 100.0% | 1322.2 |
| register | 1 | 0.0% | 1322.0 |
| search_advanced | 2 | 0.0% | 1322.0 |
