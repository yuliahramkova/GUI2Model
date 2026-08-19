# Target GUI multi-step baseline

- created_at: `2026-08-19T09:30:12.211342+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (cua)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/tasks.json`
- base_url: `http://localhost:7770`
- tasks: 10
- rag_config: `{"type": "rag", "mode": "cua", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

## Summary

| metric | value |
|---|---:|
| success_rate | 60.0% |
| n_success | 6 / 10 |
| avg_steps | 3.4 |
| avg_expected_steps | 2.3 |
| avg_steps_delta | +1.1 |
| avg_steps (success only) | 2.3 |
| total_tokens | 290411 |
| avg_tokens | 29041.1 |

## Per task

| id | success | steps | expected | delta | tokens |
|---|---|---:|---:|---:|---:|
| eval_search_shirt_results | no | 3 | 3 | +0 | 27160 |
| eval_open_create_account | yes | 1 | 1 | +0 | 9121 |
| eval_forgot_password_from_login | yes | 1 | 1 | +0 | 7145 |
| eval_browse_video_games | yes | 3 | 1 | +2 | 27825 |
| eval_product_add_wishlist | no | 4 | 2 | +2 | 30167 |
| eval_empty_cart_from_home | no | 3 | 2 | +1 | 27512 |
| eval_advanced_search_fill_name | yes | 5 | 5 | +0 | 40351 |
| eval_login_then_address_book | no | 10 | 4 | +6 | 84537 |
| eval_category_to_headphones | yes | 2 | 1 | +1 | 18304 |
| eval_minicart_to_full_cart | yes | 2 | 3 | -1 | 18289 |
