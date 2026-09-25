"""
A small enterprise knowledge base used by the embeddings and RAG examples.

It's deliberately built to exercise the failure modes that bite real retrieval systems:

* **Hard negatives.** `it-001` (how to *reset* your password) and `it-002`
  (password *policy*) share a topic but answer different questions.
  Topic-matching retrievers confuse them.
* **Exact identifiers.** `it-004` is about error `ERR-4012`, and `it-005`
  about `ERR-4013`. Dense embeddings blur these; BM25 nails them.
* **Synonyms.** A query about "automobile reimbursement" should find the
  "car mileage expense" doc with no shared words. BM25 misses it; dense
  retrieval finds it.
* **Access control.** Some docs carry `acl` groups (e.g. only `finance` or
  `hr` may see them) for permission-aware retrieval in `primer.agents.rag`.
* **Freshness.** `updated` dates let you demo metadata filtering and stale
  content (e.g. two versions of the travel policy).

`LABELED_QUERIES` maps realistic questions to the doc IDs that answer them,
the "golden set" for recall@k / MRR / nDCG in
`primer.ml.metrics` and `primer.ml.embeddings.retrieval`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str
    department: str
    updated: str  # ISO date
    acl: frozenset[str] = field(default_factory=lambda: frozenset({"everyone"}))


DOCS: list[Doc] = [
    Doc("it-001", "How to reset your password",
        "If you forgot your password or your account is locked, go to the self-service portal, "
        "choose 'Forgot password', verify with your authenticator app, and set a new password. "
        "The reset link expires after 15 minutes.",
        "IT", "2026-03-02"),
    Doc("it-002", "Password policy",
        "Passwords must be at least 14 characters and include a number and a symbol. "
        "Passwords expire every 180 days and the last 10 passwords cannot be reused. "
        "This policy applies to all employees and contractors.",
        "IT", "2025-11-20"),
    Doc("it-003", "Setting up the VPN",
        "Install the AnyConnect client from the software center, sign in with your company credentials, "
        "and approve the two-factor prompt. Use the VPN for all remote access to internal systems.",
        "IT", "2026-01-15"),
    Doc("it-004", "Error ERR-4012: VPN tunnel failed",
        "ERR-4012 means the VPN tunnel could not be established, usually because the client is out of date. "
        "Update AnyConnect to version 5.1 or later and reboot.",
        "IT", "2026-04-10"),
    Doc("it-005", "Error ERR-4013: certificate expired",
        "ERR-4013 means your device certificate expired. Open the software center and run "
        "'Renew device certificate', then reconnect.",
        "IT", "2026-04-10"),
    Doc("it-006", "Setting up two-factor authentication",
        "Download the authenticator app, scan the QR code on the security page, and enter the six digit code "
        "to finish enrolling in MFA. Two-factor is required for email and VPN.",
        "IT", "2025-09-01"),
    Doc("it-007", "Reporting phishing emails",
        "If an email looks suspicious, do not click links. Use the 'Report phishing' button in Outlook. "
        "Security reviews every report within one business day.",
        "IT", "2026-02-11"),
    Doc("it-008", "Requesting a new laptop",
        "Laptops are refreshed every three years. Submit a hardware request in the IT portal with your "
        "manager's approval. Standard devices are MacBook Pro or ThinkPad X1.",
        "IT", "2025-12-05"),
    Doc("it-009", "Printer troubleshooting",
        "If printing fails, check the toner and paper tray, then remove and re-add the printer from settings. "
        "Floor printers are named by building and floor.",
        "IT", "2024-06-30"),
    Doc("fin-001", "Car mileage expense",
        "Employees who use a personal vehicle for business driving are reimbursed at the standard mileage "
        "rate. Log each trip with date, distance and purpose, and submit the expense within 30 days.",
        "Finance", "2026-01-05"),
    Doc("fin-002", "Travel policy (2026)",
        "Book flights and hotels through the travel portal. Economy class is required for flights under "
        "six hours. Meals are covered by a daily per-diem of 75 dollars.",
        "Finance", "2026-01-01"),
    Doc("fin-003", "Travel policy (2023, superseded)",
        "Book flights through the travel agency by phone. Business class is allowed for flights over "
        "four hours. Meals are covered by a daily per-diem of 60 dollars.",
        "Finance", "2023-01-01"),
    Doc("fin-004", "Submitting expense receipts",
        "Upload receipts to the expense tool within 30 days. Receipts are required for any expense "
        "over 25 dollars. Reimbursements are paid with the next payroll run.",
        "Finance", "2025-10-12"),
    Doc("fin-005", "Vendor invoice reconciliation",
        "At quarter end, match each vendor invoice to its payment record. List every mismatch with the "
        "invoice number, amount and vendor, and send the reconciliation to the controller.",
        "Finance", "2026-03-31", frozenset({"finance"})),
    Doc("fin-006", "Q3 revenue forecast",
        "Q3 revenue is forecast at 41 million dollars, up 12 percent, driven by enterprise renewals.",
        "Finance", "2026-07-01", frozenset({"finance", "exec"})),
    Doc("hr-001", "Paid time off",
        "Full-time employees accrue 20 days of PTO per year. Request vacation in the HR portal at least "
        "two weeks ahead. Unused PTO up to 5 days rolls over.",
        "HR", "2026-01-01"),
    Doc("hr-002", "Sick leave",
        "Employees receive 10 paid sick days per year. No manager approval is needed for sick leave, "
        "but notify your team as early as possible.",
        "HR", "2025-08-15"),
    Doc("hr-003", "New hire onboarding",
        "On your first day, collect your laptop from IT, complete security training, and set up two-factor "
        "authentication. Your manager will schedule orientation sessions for week one.",
        "HR", "2026-02-01"),
    Doc("hr-004", "Salary bands and bonus",
        "Salary bands are reviewed every April. The annual bonus target is 10 percent of base pay, "
        "paid in March based on company and individual performance.",
        "HR", "2026-04-01", frozenset({"hr", "exec"})),
    Doc("hr-005", "Parental leave",
        "Primary caregivers receive 16 weeks of paid parental leave; secondary caregivers receive 6 weeks. "
        "Leave can start up to two weeks before the expected birth or adoption date.",
        "HR", "2025-05-20"),
]

DOCS_BY_ID: dict[str, Doc] = {d.id: d for d in DOCS}


# (query, set of relevant doc ids). The retrieval "golden set".
LABELED_QUERIES: list[tuple[str, set[str]]] = [
    ("I forgot my password and I'm locked out", {"it-001"}),
    ("how long must a password be", {"it-002"}),
    ("what does ERR-4012 mean", {"it-004"}),
    ("ERR-4013 on my laptop", {"it-005"}),
    ("automobile reimbursement for business driving", {"fin-001"}),
    ("how many vacation days do I get", {"hr-001"}),
    ("connect to internal systems from home", {"it-003"}),
    ("suspicious email with a link", {"it-007"}),
    ("per-diem for meals when traveling", {"fin-002"}),
    ("enroll in MFA", {"it-006"}),
    ("first day checklist for a new hire", {"hr-003"}),
    ("match vendor bills to payments", {"fin-005"}),
]
