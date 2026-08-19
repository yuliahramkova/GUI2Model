# Target GUI multi-step baseline

- created_at: `2026-08-19T15:53:37.593215+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (cua_docs)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/tasks.json`
- base_url: `http://localhost:7770`
- tasks: 10
- rag_config: `{"type": "rag", "mode": "cua_docs", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

## Summary

| metric | value |
|---|---:|
| success_rate | 50.0% |
| n_success | 5 / 10 |
| avg_steps | 3.1 |
| avg_expected_steps | 2.3 |
| avg_steps_delta | +0.8 |
| avg_steps (success only) | 1.6 |
| total_tokens | 272559 |
| avg_tokens | 27255.9 |

## Per task

| id | success | steps | expected | delta | tokens |
|---|---|---:|---:|---:|---:|
| eval_search_shirt_results | no | 3 | 3 | +0 | 26937 |
| eval_open_create_account | yes | 1 | 1 | +0 | 8844 |
| eval_forgot_password_from_login | yes | 1 | 1 | +0 | 7392 |
| eval_browse_video_games | yes | 1 | 1 | +0 | 8884 |
| eval_product_add_wishlist | no | 4 | 2 | +2 | 30013 |
| eval_empty_cart_from_home | no | 3 | 2 | +1 | 28237 |
| eval_advanced_search_fill_name | no | 3 | 5 | -2 | 27053 |
| eval_login_then_address_book | no | 10 | 4 | +6 | 89702 |
| eval_category_to_headphones | yes | 2 | 1 | +1 | 18054 |
| eval_minicart_to_full_cart | yes | 3 | 3 | +0 | 27443 |
