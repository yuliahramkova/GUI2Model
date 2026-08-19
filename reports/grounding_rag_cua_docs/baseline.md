# Target GUI grounding baseline

- created_at: `2026-08-19T09:52:50.285517+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (cua_docs)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/grounding.json`
- samples: 32
- rag_config: `{"type": "rag", "mode": "cua_docs", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |
|---|---:|---:|---:|---:|---:|
| overall | 32 | 78.1% | 43285 | 1352.7 | 0 |

## By screen

| screen | n | accuracy | avg_tokens |
|---|---:|---:|---:|
| account_logged_in | 3 | 66.7% | 1350.7 |
| cart_with_item | 3 | 100.0% | 1352.3 |
| category | 1 | 100.0% | 1353.0 |
| contact | 2 | 0.0% | 1353.0 |
| forgot_password | 1 | 100.0% | 1350.0 |
| home | 9 | 77.8% | 1353.3 |
| login | 3 | 100.0% | 1352.7 |
| minicart_open | 2 | 100.0% | 1357.0 |
| orders_returns | 1 | 100.0% | 1350.0 |
| product | 4 | 100.0% | 1352.2 |
| register | 1 | 0.0% | 1352.0 |
| search_advanced | 2 | 50.0% | 1352.0 |
