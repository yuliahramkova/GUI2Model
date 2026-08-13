# Target GUI multi-step baseline

- created_at: `2026-08-13T14:45:39.633048+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **none** (no RAG / no LoRA)
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/tasks.json`
- base_url: `http://localhost:7770`
- tasks: 10

## Summary

| metric | value |
|---|---:|
| success_rate | 40.0% |
| n_success | 4 / 10 |
| avg_steps | 2.5 |
| avg_expected_steps | 2.3 |
| avg_steps_delta | +0.2 |
| avg_steps (success only) | 1.5 |
| total_tokens | 202662 |
| avg_tokens | 20266.2 |

## Per task

| id | success | steps | expected | delta | tokens |
|---|---|---:|---:|---:|---:|
| eval_search_shirt_results | no | 3 | 3 | +0 | 25918 |
| eval_open_create_account | yes | 1 | 1 | +0 | 8541 |
| eval_forgot_password_from_login | yes | 1 | 1 | +0 | 6688 |
| eval_browse_video_games | yes | 2 | 1 | +1 | 17150 |
| eval_product_add_wishlist | no | 4 | 2 | +2 | 28621 |
| eval_empty_cart_from_home | no | 3 | 2 | +1 | 25947 |
| eval_advanced_search_fill_name | no | 3 | 5 | -2 | 25933 |
| eval_login_then_address_book | no | 3 | 4 | -1 | 20467 |
| eval_category_to_headphones | yes | 2 | 1 | +1 | 17285 |
| eval_minicart_to_full_cart | no | 3 | 3 | +0 | 26112 |
