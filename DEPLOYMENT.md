# Deployment

## Git

```bash
git remote add origin git@github.com:justinmaccabe/codex_terminal.git
git add .
git commit -m "Initial codex_terminal app"
git push -u origin main
```

## Snowflake

This app uses external market-data requests, so Snowflake deployment needs:

- a `STREAMLIT` app object
- an external access integration
- host allowlisting for Yahoo Finance, FRED, and Vanguard

Use [snowflake.sql](/Users/optadmin/Documents/Playground/snowflake.sql) as the starting template.

If you deploy with Snowflake CLI, you will also need local Snowflake auth configured first.
