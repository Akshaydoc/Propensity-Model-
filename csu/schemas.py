"""Expected schemas for every source table (used by batch ingestion + schema checks)."""
S, N, D = "str", "num", "date"

SCHEMAS = {
    "dim_customer": dict(key=["customer_number"], not_null=["customer_number", "onboarding_date"],
        cols={"customer_number": S, "onboarding_date": D, "incorporation_date": D, "industry_desc": S,
              "sector_code": S, "cust_llc_flag": S, "cust_sub_segment": S, "rm_id": S}),
    "sme_portfolio": dict(key=["account_no", "timekey"], not_null=["account_no", "customer_number", "timekey", "product_type"],
        cols={"account_no": S, "customer_number": S, "timekey": N, "product_type": S, "gl_no": S, "exclusion_1": S}),
    "fact_balance_sheet": dict(key=["contract_id", "time_key"], not_null=["contract_id", "time_key"],
        cols={"time_key": N, "contract_id": S, "acc_deal_number": S, "gl_no": S, "currency_key": N,
              "deal_balance": N, "category_code": S}),
    "dim_accounts": dict(key=["account_number"], not_null=["account_number", "customer_number", "account_open_date"],
        cols={"account_number": S, "customer_number": S, "account_open_date": D, "closure_date": D,
              "account_status": S, "latest_flag": S}),
    "casa_avg_balance": dict(key=["account_number", "period_date"], not_null=["account_number", "period_date"],
        cols={"account_number": S, "period_date": D, "monthly_average_balance": N, "closing_balance": N}),
    "loan_snapshot": dict(key=["contract_id", "timekey"], not_null=["contract_id", "timekey"],
        cols={"contract_id": S, "timekey": N, "dpd": N, "outstanding": N, "tenor_months": N}),
    "casa_txn_monthly": dict(key=["customer_number", "timekey", "txn_type"], not_null=["customer_number", "timekey"],
        cols={"customer_number": S, "timekey": N, "txn_type": S, "txn_cnt": N, "txn_amt": N}),
    "cc_txn_monthly": dict(key=["customer_number", "timekey"], not_null=["customer_number", "timekey"],
        cols={"customer_number": S, "timekey": N, "logo_type": S, "txn_cnt": N, "txn_amt": N, "cash_amt": N}),
    "dim_currency": dict(key=["currency_key"], not_null=["currency_key"], cols={"currency_key": N, "ccy_id": S}),
    "eligibility_policy": dict(key=["product"], not_null=["product"],
        cols={"product": S, "min_lor_months": N, "min_cust_age": N, "allowed_segments": S, "blocked_industries": S}),
    "product_revenue": dict(key=["product"], not_null=["product"], cols={"product": S, "margin_rate": N, "avg_ticket": N}),
}
# sanity ranges (rows outside are quarantined)
RANGES = {("fact_balance_sheet", "deal_balance"): (-1e10, 1e10), ("loan_snapshot", "dpd"): (0, 5000),
          ("casa_txn_monthly", "txn_amt"): (0, 1e12), ("cc_txn_monthly", "txn_amt"): (0, 1e12)}
