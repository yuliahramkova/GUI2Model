# Target GUI multi-step baseline

- created_at: `2026-08-19T12:21:38.565057+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (docs)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/tasks.json`
- base_url: `http://localhost:7770`
- tasks: 10
- rag_config: `{"type": "rag", "mode": "docs", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

## Summary

| metric | value |
|---|---:|
| success_rate | 60.0% |
| n_success | 6 / 10 |
| avg_steps | 2.7 |
| avg_expected_steps | 2.3 |
| avg_steps_delta | +0.4 |
| avg_steps (success only) | 2.3 |
| total_tokens | 229055 |
| avg_tokens | 22905.5 |

## Per task

| id | success | steps | expected | delta | tokens |
|---|---|---:|---:|---:|---:|
| eval_search_shirt_results | no | 3 | 3 | +0 | 27164 |
| eval_open_create_account | yes | 1 | 1 | +0 | 8967 |
| eval_forgot_password_from_login | yes | 1 | 1 | +0 | 7043 |
| eval_browse_video_games | yes | 1 | 1 | +0 | 8795 |
| eval_product_add_wishlist | no | 4 | 2 | +2 | 30362 |
| eval_empty_cart_from_home | no | 3 | 2 | +1 | 26998 |
| eval_advanced_search_fill_name | no | 3 | 5 | -2 | 27421 |
| eval_login_then_address_book | yes | 7 | 4 | +3 | 55492 |
| eval_category_to_headphones | yes | 2 | 1 | +1 | 18046 |
| eval_minicart_to_full_cart | yes | 2 | 3 | -1 | 18767 |
