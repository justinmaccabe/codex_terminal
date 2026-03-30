-- codex_terminal Snowflake setup template
-- Update the placeholder identifiers before running.

-- 1. Create or use a database/schema/warehouse.
-- use role <your_role>;
-- create database if not exists CODEX_TERMINAL;
-- create schema if not exists CODEX_TERMINAL.APP;
-- create warehouse if not exists CODEX_TERMINAL_WH warehouse_size = 'SMALL' auto_suspend = 60;

-- 2. External network access for live market data.
create or replace network rule codex_terminal_egress_rule
  mode = egress
  type = host_port
  value_list = (
    'query1.finance.yahoo.com',
    'finance.yahoo.com',
    'yfapi.net',
    'api.stlouisfed.org',
    'fred.stlouisfed.org',
    'investor.vanguard.com'
  );

create or replace external access integration codex_terminal_external_access
  allowed_network_rules = (codex_terminal_egress_rule)
  enabled = true;

-- 3. Create the Streamlit app.
-- For warehouse runtime, environment.yml is used. For container runtime,
-- update this section to match your account setup and compute pool choices.
create or replace streamlit codex_terminal
  root_location = '@CODEX_TERMINAL.APP.CODEX_STAGE'
  main_file = '/app.py'
  query_warehouse = CODEX_TERMINAL_WH
  title = 'codex_terminal'
  external_access_integrations = (codex_terminal_external_access);

