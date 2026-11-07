import sqlite3
import json
from decimal import Decimal
from datetime import date
import app

def test_integration():
    doc_id = "badf1f954060a358"
    with app.get_db_connection() as conn:
        existing = conn.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if not existing:
        print("Indexing test_docs/complex_accounts_receivable_transaction_book.pdf into accounting.db...")
        with open("test_docs/complex_accounts_receivable_transaction_book.pdf", "rb") as f:
            pdf_bytes = f.read()
        emb_model = app.get_embedding_model()
        ok, msg, indexed_id = app.index_document(
            "complex_accounts_receivable_transaction_book.pdf",
            pdf_bytes,
            emb_model
        )
        assert ok, f"Failed to index test PDF: {msg}"
        doc_id = indexed_id

    active_docs = [doc_id]
    as_of = date(2026, 11, 6)

    print("--- 1. Testing Receivables by Currency ---")
    rc = app.calculate_receivables_by_currency(active_docs, as_of_date=as_of)
    aed = rc["currencies"]["AED"]
    usd = rc["currencies"]["USD"]
    print("AED Gross:", aed["gross_outstanding"])
    print("AED Unallocated:", aed["unallocated_cash"])
    print("AED Net:", aed["net_exposure"])
    print("USD Gross:", usd["gross_outstanding"])
    print("USD Unallocated:", usd["unallocated_cash"])
    print("USD Net:", usd["net_exposure"])
    assert aed["gross_outstanding"] == "153,144.78"
    assert aed["unallocated_cash"] == "4,250.00"
    assert aed["net_exposure"] == "148,894.78"
    assert aed["original_face_total"] == "234,807.86"
    assert aed["applied_cash_total"] == "78,369.06"
    assert aed["applied_credits_total"] == "3,294.02"

    assert usd["gross_outstanding"] == "37,090.34"
    assert usd["unallocated_cash"] == "300.00"
    assert usd["net_exposure"] == "36,790.34"
    assert usd["original_face_total"] == "63,344.15"
    assert usd["applied_cash_total"] == "25,965.69"
    assert usd["applied_credits_total"] == "288.12"
    print("✓ Receivables by Currency test passed!")

    print("\n--- 2. Testing Harborline Exposure ---")
    hl = app.calculate_customer_exposure("Harborline", active_docs, as_of_date=as_of)
    hl_aed = hl["by_currency"]["AED"]
    hl_usd = hl["by_currency"]["USD"]
    print("Harborline AED Gross:", hl_aed["gross_outstanding"])
    print("Harborline AED Unallocated:", hl_aed["unallocated_cash"])
    print("Harborline AED Net:", hl_aed["net_exposure"])
    print("Harborline USD Gross:", hl_usd["gross_outstanding"])
    print("Harborline USD Unallocated:", hl_usd["unallocated_cash"])
    print("Harborline USD Net:", hl_usd["net_exposure"])
    assert hl_aed["gross_outstanding"] == "99,924.15"
    assert hl_aed["unallocated_cash"] == "2,500.00"
    assert hl_aed["net_exposure"] == "97,424.15"
    assert hl_usd["gross_outstanding"] == "23,878.65"
    assert hl_usd["unallocated_cash"] == "0.00"
    assert hl_usd["net_exposure"] == "23,878.65"
    print("✓ Harborline Exposure test passed!")

    print("\n--- 3. Testing Meridian Exposure ---")
    mr = app.calculate_customer_exposure("Meridian", active_docs, as_of_date=as_of)
    mr_aed = mr["by_currency"]["AED"]
    mr_usd = mr["by_currency"]["USD"]
    print("Meridian AED Gross:", mr_aed["gross_outstanding"])
    print("Meridian AED Unallocated:", mr_aed["unallocated_cash"])
    print("Meridian AED Net:", mr_aed["net_exposure"])
    print("Meridian USD Gross:", mr_usd["gross_outstanding"])
    print("Meridian USD Unallocated:", mr_usd["unallocated_cash"])
    print("Meridian USD Net:", mr_usd["net_exposure"])
    assert mr_aed["gross_outstanding"] == "53,220.63"
    assert mr_aed["unallocated_cash"] == "1,750.00"
    assert mr_aed["net_exposure"] == "51,470.63"
    assert mr_usd["gross_outstanding"] == "13,211.69"
    assert mr_usd["unallocated_cash"] == "300.00"
    assert mr_usd["net_exposure"] == "12,911.69"
    print("✓ Meridian Exposure test passed!")

    print("\n--- 4. Testing Overdue Invoices as of 2026-11-06 ---")
    od = app.evaluate_overdue_invoices(active_docs, as_of_date=as_of)
    print("Overdue AED:", od["overdue_totals_by_currency"]["AED"])
    print("Overdue USD:", od["overdue_totals_by_currency"]["USD"])
    print("Confirmed overdue count:", len(od["confirmed_overdue"]))
    assert od["overdue_totals_by_currency"]["AED"] == "100,599.27"
    assert od["overdue_totals_by_currency"]["USD"] == "28,779.49"
    assert len(od["confirmed_overdue"]) == 20
    unknown_ids = [u["invoice_number"] for u in od["unclassified_unknown_due_date"]]
    assert "AR-0013" in unknown_ids
    settled_ids = [s["invoice_number"] for s in od["settled"]]
    assert "AR-0028" in settled_ids
    print("✓ Overdue evaluation test passed!")

    print("\n--- 5. Testing Highest Invoice Breakdown ---")
    hi = app.find_highest_invoice(active_docs)
    print("Highest AED Face:", hi["by_currency"]["AED"]["highest_face"]["invoice_number"], hi["by_currency"]["AED"]["highest_face"]["amount"])
    print("Highest AED Open:", hi["by_currency"]["AED"]["highest_open"]["invoice_number"], hi["by_currency"]["AED"]["highest_open"]["amount"])
    print("Highest USD Face:", hi["by_currency"]["USD"]["highest_face"]["invoice_number"], hi["by_currency"]["USD"]["highest_face"]["amount"])
    print("Highest USD Open:", hi["by_currency"]["USD"]["highest_open"]["invoice_number"], hi["by_currency"]["USD"]["highest_open"]["amount"])
    assert hi["by_currency"]["AED"]["highest_face"]["invoice_number"] == "AR-0022"
    assert hi["by_currency"]["AED"]["highest_face"]["amount"] == "13,291.06"
    assert hi["by_currency"]["AED"]["highest_open"]["invoice_number"] == "AR-0022"
    assert hi["by_currency"]["AED"]["highest_open"]["amount"] == "13,291.06"
    assert hi["by_currency"]["USD"]["highest_face"]["invoice_number"] == "AR-0040"
    assert hi["by_currency"]["USD"]["highest_face"]["amount"] == "12,899.70"
    assert hi["by_currency"]["USD"]["highest_open"]["invoice_number"] == "AR-0005"
    assert hi["by_currency"]["USD"]["highest_open"]["amount"] == "9,812.72"
    print("✓ Highest Invoice test passed!")

    print("\n--- 6. Testing Exact Lookups (AR-0035, AR-0003, AR-0021) ---")
    inv35 = app.lookup_single_invoice("AR-0035", active_docs)
    assert inv35["found"] == True
    assert inv35["original_total"] == "2881.18"
    assert inv35["applied_credits"] == "288.12"
    assert inv35["posted_outstanding"] == "2593.06"
    assert inv35["disputed_amount"] == "500.00"
    assert inv35["unverified_claim"] == "2593.06"
    assert set([2, 40, 49, 50]).issubset(set(inv35["source_pages"]))

    inv3 = app.lookup_single_invoice("AR-0003", active_docs)
    assert inv3["found"] == True
    assert inv3["original_total"] == "10243.67"
    assert inv3["posted_outstanding"] == "6658.39"
    assert inv3["applied_payment"] == "3585.28"

    inv21 = app.lookup_single_invoice("AR-0021", active_docs)
    assert inv21["found"] == True
    assert inv21["original_total"] == "10496.09"
    assert inv21["applied_payment"] == "3306.27"
    assert inv21["applied_credits"] == "1049.61"
    assert inv21["posted_outstanding"] == "6140.21"
    print("✓ Exact lookups passed!")

    print("\n--- 7. Testing Question Routing for All 18 Questions ---")
    with open("accounting_ground_truth.json") as f:
        gt = json.load(f)

    for idx, item in enumerate(gt["test_questions"]):
        q = item["question"]
        route, args = app.route_question(q, active_docs)
        print(f"  Q{idx+1}: {route:25s} | {q}")
        assert route != "UNKNOWN"
        if "AR-0035" in q or "AR-0021" in q:
            assert route == "SINGLE_INVOICE_LOOKUP"
        elif "overdue" in q.lower():
            assert route == "OVERDUE_INVOICES"
        elif "harborline" in q.lower() and "exposure" in q.lower():
            assert route == "CUSTOMER_NET_EXPOSURE"
        elif "meridian" in q.lower() and "exposure" in q.lower():
            assert route == "CUSTOMER_NET_EXPOSURE"
        elif "disputed" in q.lower():
            assert route == "DISPUTED_BALANCES_AUDIT"
        elif "combined" in q.lower():
            assert route == "COMBINED_CURRENCY"
        elif "bank" in q.lower():
            assert route == "BANK_ACCOUNT"

    print("✓ All 18 question routing checks passed!")

    print("\n--- 8. Testing Citation Validation ---")
    valid_set = {"S1", "S2", "S3"}
    test_ans = "According to [S1] and [S2], the balance is AED 7,600. [S99] is fake."
    sanitized, valid, invalid = app.validate_source_ids(test_ans, valid_set)
    assert "[Unverified Source S99]" in sanitized
    assert valid == ["S1", "S2"]
    assert invalid == ["S99"]
    print("✓ Citation validation test passed!")

    print("\n--- 9. Testing Format Headline Output ---")
    hl_headline = app.format_calculation_headline(hl)
    assert "AED 99,924.15" in hl_headline
    assert "AED 2,500.00" in hl_headline
    assert "AED 97,424.15" in hl_headline
    assert "USD 23,878.65" in hl_headline
    print("✓ Format Headline Output test passed!")

    print("\n--- 10. Testing Document Scope Filtering ---")
    empty_res = app.calculate_receivables_by_currency([])
    assert len(empty_res["currencies"]) == 0
    print("✓ Document Scope Filtering test passed!")

    print("\n--- 11. Testing Dynamic Audit Calculations & High-Context Ledger Summary ---")
    vc = app.calculate_vendor_customer_counterparties(active_docs)
    assert vc["vendors"] == ["CedarBridge Systems FZ-LLC"]
    assert vc["customers"] == ["Harborline Manufacturing LLC", "Meridian Retail Group LLC"]

    cnts = app.calculate_record_counts(active_docs)
    assert cnts["unique_invoices"] == 40
    assert cnts["priced_line_items"] == 160

    disp = app.calculate_disputed_balances(active_docs)
    disp_ids = [d["invoice_number"] for d in disp["disputed_invoices"]]
    assert disp_ids == ["AR-0007", "AR-0019", "AR-0035"]

    unk = app.calculate_unknown_due_dates(active_docs)
    unk_ids = [u["invoice_number"] for u in unk["unknown_due_invoices"]]
    assert unk_ids == ["AR-0013", "AR-0028"]

    hline = app.calculate_highest_line_items(active_docs)
    assert hline["highest_by_currency"]["AED"]["invoice_number"] == "AR-0022"
    assert hline["highest_by_currency"]["AED"]["total"] == "5,112.45"
    assert hline["highest_by_currency"]["USD"]["invoice_number"] == "AR-0015"
    assert hline["highest_by_currency"]["USD"]["total"] == "4,919.40"

    ledger_sum = app.build_ledger_summary_for_prompt(active_docs)
    assert "[INVOICE AR-0001]" in ledger_sum
    assert "[INVOICE AR-0040]" in ledger_sum
    assert "[UNALLOCATED_CASH UC-001]" in ledger_sum
    print("✓ Dynamic Audit Calculations & High-Context Ledger Summary passed!")

    print("\n==========================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    test_integration()
