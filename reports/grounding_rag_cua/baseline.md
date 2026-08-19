# Target GUI grounding baseline

- created_at: `2026-08-19T04:09:54.165336+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (cua)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/grounding.json`
- samples: 32
- rag_config: `{"type": "rag", "mode": "cua", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |
|---|---:|---:|---:|---:|---:|
| overall | 32 | 78.1% | 56482 | 1765.1 | 0 |

## By screen

| screen | n | accuracy | avg_tokens |
|---|---:|---:|---:|
| account_logged_in | 3 | 66.7% | 1791.3 |
| cart_with_item | 3 | 100.0% | 1768.0 |
| category | 1 | 100.0% | 2948.0 |
| contact | 2 | 50.0% | 1744.5 |
| forgot_password | 1 | 100.0% | 1734.0 |
| home | 9 | 66.7% | 1602.7 |
| login | 3 | 66.7% | 1849.0 |
| minicart_open | 2 | 100.0% | 1873.0 |
| orders_returns | 1 | 100.0% | 1589.0 |
| product | 4 | 100.0% | 1786.8 |
| register | 1 | 100.0% | 1767.0 |
| search_advanced | 2 | 50.0% | 1706.5 |
