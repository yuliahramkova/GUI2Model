# Target GUI grounding baseline

- created_at: `2026-08-19T12:08:40.038667+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (docs)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/grounding.json`
- samples: 32
- rag_config: `{"type": "rag", "mode": "docs", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |
|---|---:|---:|---:|---:|---:|
| overall | 32 | 78.1% | 52569 | 1642.8 | 0 |

## By screen

| screen | n | accuracy | avg_tokens |
|---|---:|---:|---:|
| account_logged_in | 3 | 66.7% | 1633.0 |
| cart_with_item | 3 | 100.0% | 1656.3 |
| category | 1 | 100.0% | 1636.0 |
| contact | 2 | 50.0% | 1617.0 |
| forgot_password | 1 | 100.0% | 1543.0 |
| home | 9 | 66.7% | 1631.4 |
| login | 3 | 100.0% | 1574.0 |
| minicart_open | 2 | 100.0% | 1722.0 |
| orders_returns | 1 | 100.0% | 1610.0 |
| product | 4 | 100.0% | 1706.8 |
| register | 1 | 0.0% | 1625.0 |
| search_advanced | 2 | 50.0% | 1688.5 |
