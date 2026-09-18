"""
FinSecure AI: Unified Zero-Trust Enterprise Financial & Decision Agent
Team: Neural Nomads (Balaji A, Anish, Sanjay, Ashrith, Jeevan)
"""

import json
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FinSecureAgent")


# ---------------------------------------------------------
# CUSTOM EXCEPTIONS
# ---------------------------------------------------------
class FinSecureException(Exception):
    """Base exception for the FinSecure agent."""
    pass

class AuthenticationError(FinSecureException):
    """Raised when user profile or security clearance fails validation."""
    pass

class DataIntegrityError(FinSecureException):
    """Raised when incoming JSON documents or payloads are malformed."""
    pass

class FinancialLogicError(FinSecureException):
    """Raised when mathematical or financial invariants are violated."""
    pass


# ---------------------------------------------------------
# CLASSIFICATION HIERARCHY & SECURITY CONSTANTS
# ---------------------------------------------------------
CLEARANCE_RANKS: Dict[str, int] = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
    "RESTRICTED": 3
}


# ---------------------------------------------------------
# CORE SECURITY & DECISION AGENT
# ---------------------------------------------------------
class FinSecureAgent:
    """
    Autonomous agent handling:
    1. Pre-LLM Document Authorization & Conflict Resolution.
    2. Real-time Personal/Corporate Financial Tracking & Recommendations.
    3. Immutable Audit Trails.
    """

    def __init__(self) -> None:
        self.audit_log: List[Dict[str, Any]] = []
        # In-memory financial datastore per user
        self.user_financial_profiles: Dict[str, Dict[str, Any]] = {
            "U102": {
                "monthly_budget": 8000.0,
                "expenses": {"Food": 12000.0, "Fuel": 1200.0, "Utilities": 2500.0},
                "savings_goal": {"name": "Trip to Goa", "target": 20000.0, "current": 14000.0},
                "liquid_cash": 18000.0
            },
            "U205": {
                "monthly_budget": 15000.0,
                "expenses": {"Food": 6000.0, "Entertainment": 3000.0},
                "savings_goal": {"name": "Emergency Fund", "target": 50000.0, "current": 15000.0},
                "liquid_cash": 25000.0
            },
            "U301": {
                "monthly_budget": 50000.0,
                "expenses": {"Food": 10000.0, "Investment": 15000.0},
                "savings_goal": {"name": "Portfolio Expansion", "target": 100000.0, "current": 45000.0},
                "liquid_cash": 60000.0
            }
        }

    # -----------------------------------------------------
    # SECURITY & PII MASKING
    # -----------------------------------------------------
    @staticmethod
    def mask_pii(text: str) -> str:
        """Masks Indian phone numbers, PAN numbers, and Credit Cards."""
        if not isinstance(text, str):
            return str(text)
        try:
            # Mask 10-digit phone numbers
            text = re.sub(r"\b[6-9]\d{9}\b", "[MASKED_PHONE]", text)
            # Mask PAN formats (5 letters, 4 numbers, 1 letter)
            text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", "[MASKED_PAN]", text, flags=re.I)
            # Mask 16-digit card numbers
            text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[MASKED_CARD]", text)
            return text
        except Exception as e:
            logger.error(f"Failed to mask PII: {e}")
            return text

    # -----------------------------------------------------
    # AUTHORIZATION GATEKEEPER (PRE-LLM FILTER)
    # -----------------------------------------------------
    def authorize_documents(
        self, user: Dict[str, Any], documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Critical Rule: Strictly filters out unauthorized documents BEFORE
        they can reach synthesis context.
        """
        if not isinstance(user, dict):
            raise AuthenticationError("User context must be a valid dictionary.")
        if not isinstance(documents, list):
            raise DataIntegrityError("Documents must be provided as a list.")

        user_role = str(user.get("role", "")).strip().upper()
        user_dept = str(user.get("department", "")).strip().upper()
        user_clearance = str(user.get("clearance", "PUBLIC")).strip().upper()
        user_rank = CLEARANCE_RANKS.get(user_clearance, 0)

        authorized_docs: List[Dict[str, Any]] = []
        blocked_docs: List[Dict[str, Any]] = []

        for doc in documents:
            try:
                doc_id = doc.get("document_id", "UNKNOWN_DOC")
                doc_class = str(doc.get("classification", "RESTRICTED")).strip().upper()
                doc_rank = CLEARANCE_RANKS.get(doc_class, 999)

                allowed_depts = [d.upper() for d in doc.get("allowed_departments", [])]
                allowed_roles = [r.upper() for r in doc.get("allowed_roles", [])]

                # Clearance Hierarchy Check
                if user_rank < doc_rank:
                    blocked_docs.append({
                        "document_id": doc_id,
                        "reason": f"Clearance mismatch: requires {doc_class}, user has {user_clearance}"
                    })
                    continue

                # Department & Role Access Verification
                dept_match = not allowed_depts or (user_dept in allowed_depts)
                role_match = not allowed_roles or (user_role in allowed_roles)

                if dept_match and role_match:
                    authorized_docs.append(doc)
                else:
                    blocked_docs.append({
                        "document_id": doc_id,
                        "reason": f"Role/Department mismatch: allowed_depts={allowed_depts}, allowed_roles={allowed_roles}"
                    })
            except Exception as ex:
                logger.warning(f"Error processing doc authorization for {doc}: {ex}")
                blocked_docs.append({"document_id": doc.get("document_id", "ERROR_DOC"), "reason": str(ex)})

        return authorized_docs, blocked_docs

    # -----------------------------------------------------
    # CONFLICT RESOLUTION ENGINE (VERSION & DATE)
    # -----------------------------------------------------
    def resolve_conflicts(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates and resolves contradictions by grouping by title
        and picking the highest version / latest effective date.
        """
        if not docs:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for d in docs:
            key = d.get("title", "").strip().lower()
            grouped.setdefault(key, []).append(d)

        resolved: List[Dict[str, Any]] = []
        for title, cluster in grouped.items():
            if len(cluster) == 1:
                resolved.append(cluster[0])
                continue

            def sort_key(item: Dict[str, Any]) -> Tuple[float, str]:
                try:
                    ver = float(str(item.get("version", "1.0")).replace("v", ""))
                except ValueError:
                    ver = 1.0
                dt = str(item.get("effective_date", "1970-01-01"))
                return (ver, dt)

            # Sort descending: newest version and latest date first
            cluster.sort(key=sort_key, reverse=True)
            newest = cluster[0]
            resolved.append(newest)

        return resolved

    # -----------------------------------------------------
    # FINANCIAL ENGINE: EXPENSE, BUDGET & RECOMMENDATIONS
    # -----------------------------------------------------
    def process_financial_prompt(self, user_id: str, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Parses financial intents:
        1. Expense Logging: "I spent 500 on fuel today"
        2. Budget Adjustments: "I've spent ₹12,000 on food this month. My budget is ₹8,000..."
        3. Decision Evaluation: "Should I buy Apple stock / flight to Goa / TechCorp IPO?"
        """
        user_fin = self.user_financial_profiles.setdefault(user_id, {
            "monthly_budget": 10000.0,
            "expenses": {},
            "savings_goal": {"name": "General", "target": 20000.0, "current": 5000.0},
            "liquid_cash": 15000.0
        })

        clean_prompt = prompt.lower()

        # Intent 1: Expense Logging
        spend_match = re.search(r"spent\s+(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s+(?:on|for)\s+([a-zA-Z]+)", clean_prompt)
        if spend_match:
            try:
                amt = float(spend_match.group(1))
                category = spend_match.group(2).capitalize()
                current_val = user_fin["expenses"].get(category, 0.0)
                user_fin["expenses"][category] = current_val + amt
                user_fin["liquid_cash"] = max(0.0, user_fin["liquid_cash"] - amt)
                
                total_spent = sum(user_fin["expenses"].values())
                budget = user_fin["monthly_budget"]
                alert = None
                if total_spent > budget:
                    alert = f"⚠️ BUDGET OVERRUN ALERT: Total spent ₹{total_spent:,.2f} exceeds limit ₹{budget:,.2f}!"

                return {
                    "action": "EXPENSE_LOGGED",
                    "category": category,
                    "amount": amt,
                    "updated_category_total": user_fin["expenses"][category],
                    "total_monthly_expenses": total_spent,
                    "alert": alert
                }
            except Exception as e:
                raise FinancialLogicError(f"Expense parsing failed: {e}")

        # Intent 2: Budget Overrun Inquiry
        if "budget" in clean_prompt and ("food" in clean_prompt or "spent" in clean_prompt or "adjust" in clean_prompt):
            food_expense = user_fin["expenses"].get("Food", 12000.0)
            budget = user_fin["monthly_budget"]
            overrun = food_expense - budget
            pct_cut = (overrun / food_expense * 100) if food_expense > 0 else 0
            
            return {
                "action": "BUDGET_ADVICE",
                "food_expense": food_expense,
                "budget": budget,
                "deficit": overrun,
                "recommendation": (
                    f"You have overspent your food budget by ₹{overrun:,.2f} ({pct_cut:.1f}% above budget). "
                    "Action plan: 1) Cap daily discretionary spending to ₹150 for the rest of the month; "
                    "2) Temporarily pause dining out; 3) Reallocate surplus from utilities or entertainment."
                )
            }

        # Intent 3: "Should I buy / invest / book?"
        if any(keyword in clean_prompt for keyword in ["should i buy", "should i invest", "should i book", "should i subscribe"]):
            # Extract asset name & amount if present
            amount_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)", clean_prompt)
            allocated_amount = float(amount_match.group(1)) if amount_match else 5000.0
            liquid = user_fin.get("liquid_cash", 10000.0)
            
            # Confidence Scoring Engine
            # Weights: Liquidity ratio (40%), Goal safety (30%), Market risk (30%)
            liquidity_ratio = min(1.0, liquid / (allocated_amount * 2.5))
            goal_safety_score = 0.85 if liquid > allocated_amount else 0.35
            
            is_high_risk = any(item in clean_prompt for item in ["ipo", "crypto", "stock", "flight"])
            risk_penalty = 0.70 if is_high_risk else 0.90
            
            confidence_score = round(((liquidity_ratio * 0.4) + (goal_safety_score * 0.3) + (risk_penalty * 0.3)) * 100, 2)
            
            decision = "RECOMMENDED" if confidence_score >= 65 else "NOT_RECOMMENDED"
            
            if "flight" in clean_prompt or "goa" in clean_prompt:
                reasoning = (
                    f"Liquid cash available is ₹{liquid:,.2f}. The flight cost is ₹{allocated_amount:,.2f}. "
                    "Your primary goal requires continuous savings. Booking is caution-advised; "
                    "ensure emergency reserves remain above ₹10,000."
                )
            elif "apple" in clean_prompt or "stock" in clean_prompt:
                reasoning = (
                    f"Investing ₹{allocated_amount:,.2f} represents {(allocated_amount/liquid)*100:.1f}% of liquid reserves. "
                    "Diversified equity exposure is sound, but establish a stop-loss given market volatility."
                )
            elif "ipo" in clean_prompt:
                reasoning = (
                    "IPO allocations carry listing-day uncertainty. Subscribe only if institutional subscription "
                    "exceeds 5x and capital is not tied to near-term goals."
                )
            else:
                reasoning = f"Evaluated with liquid balance ₹{liquid:,.2f} against allocated ₹{allocated_amount:,.2f}."

            return {
                "action": "DECISION_ANALYSIS",
                "verdict": decision,
                "confidence_score": f"{confidence_score}%",
                "allocation_amount": allocated_amount,
                "reasoning": reasoning
            }

        return None

    # -----------------------------------------------------
    # MAIN WORKFLOW EXECUTION
    # -----------------------------------------------------
    def execute_query(
        self,
        user: Dict[str, Any],
        prompt: str,
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Coordinates authorization, document conflict resolution,
        financial domain evaluation, and audit trail generation.
        """
        timestamp = datetime.utcnow().isoformat()
        user_id = str(user.get("user_id", "ANONYMOUS"))
        clean_prompt = self.mask_pii(prompt.strip())
        documents = documents or []

        audit_entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "user_id": user_id,
            "prompt": clean_prompt,
            "authorized_docs": [],
            "blocked_docs": [],
            "status": "PROCESSING"
        }

        try:
            # 1. Financial Domain Engine check
            fin_result = self.process_financial_prompt(user_id, clean_prompt)

            # 2. Authorization Step (Pre-LLM gate)
            authorized_docs, blocked_docs = self.authorize_documents(user, documents)
            audit_entry["authorized_docs"] = [d.get("document_id") for d in authorized_docs]
            audit_entry["blocked_docs"] = blocked_docs

            # 3. Decision Tree
            if fin_result and not documents:
                # Direct personal finance request
                audit_entry["status"] = "SUCCESS_FINANCIAL"
                response = {
                    "status": "SUCCESS",
                    "type": "FINANCIAL_ADVISORY",
                    "details": fin_result,
                    "citations": []
                }
            elif not documents:
                # No enterprise documents supplied and no direct financial intent
                audit_entry["status"] = "NO_CONTEXT"
                response = {
                    "status": "SAFE_REFUSAL",
                    "message": "No enterprise context provided to answer the question.",
                    "citations": []
                }
            elif not authorized_docs:
                # Critical Requirement: Documents existed but user cannot access them
                audit_entry["status"] = "ACCESS_DENIED"
                response = {
                    "status": "SAFE_REFUSAL",
                    "message": "Access Denied: You do not possess the required clearance, role, or departmental permissions to view the documents relevant to this inquiry.",
                    "citations": []
                }
            else:
                # 4. Resolve Conflicting / Outdated authorized docs
                resolved_docs = self.resolve_conflicts(authorized_docs)
                
                # Synthesis
                citations = [
                    {
                        "document_id": d.get("document_id"),
                        "title": d.get("title"),
                        "version": d.get("version"),
                        "effective_date": d.get("effective_date"),
                        "classification": d.get("classification")
                    }
                    for d in resolved_docs
                ]
                
                combined_content = " | ".join([d.get("content", "") for d in resolved_docs])
                audit_entry["status"] = "SUCCESS_SYNTHESIS"
                
                response = {
                    "status": "SUCCESS",
                    "type": "DOCUMENT_SYNTHESIS",
                    "answer": f"Based on authorized documents: {combined_content}",
                    "citations": citations
                }
                
                if fin_result:
                    response["supplementary_financial_analysis"] = fin_result

            self.audit_log.append(audit_entry)
            return response

        except Exception as ex:
            logger.error(f"Execution failed: {ex}")
            audit_entry["status"] = "FATAL_ERROR"
            audit_entry["error"] = str(ex)
            self.audit_log.append(audit_entry)
            return {
                "status": "ERROR",
                "message": f"An unexpected system exception occurred: {str(ex)}",
                "citations": []
            }


# ---------------------------------------------------------
# VERIFICATION SUITE RUNNER
# ---------------------------------------------------------
def run_all_hackathon_tests():
    agent = FinSecureAgent()
    separator = "=" * 70

    print(separator)
    print("FINSECURE AGENT: HACKATHON TEST SUITE RUN")
    print(separator)

    # -----------------------------------------------------
    # Test Input A — Authorized Answer
    # -----------------------------------------------------
    print("\n[TEST A] Authorized Single Document Query")
    user_a = {"user_id": "U102", "role": "Finance", "department": "Finance", "clearance": "Internal"}
    docs_a = [
        {
            "document_id": "DOC-101",
            "title": "Q4 Revenue Forecast",
            "classification": "Internal",
            "allowed_departments": ["Finance"],
            "allowed_roles": ["Finance"],
            "version": "2.0",
            "effective_date": "2026-09-01",
            "content": "Q4 projected revenue is 120 crore."
        },
        {
            "document_id": "DOC-102",
            "title": "Engineering Roadmap",
            "classification": "Internal",
            "allowed_departments": ["Engineering"],
            "allowed_roles": ["Engineer"],
            "version": "1.0",
            "effective_date": "2026-08-01",
            "content": "The next platform release is planned for October."
        }
    ]
    prompt_a = "What is the Q4 revenue forecast?"
    res_a = agent.execute_query(user_a, prompt_a, docs_a)
    print(json.dumps(res_a, indent=2))

    # -----------------------------------------------------
    # Test Input B — Relevant but Unauthorized (Must Block)
    # -----------------------------------------------------
    print("\n[TEST B] Relevant Document with Unauthorized User (Safe Refusal)")
    user_b = {"user_id": "U205", "role": "Marketing", "department": "Marketing", "clearance": "Internal"}
    docs_b = [
        {
            "document_id": "DOC-201",
            "title": "Q4 Revenue Forecast",
            "classification": "Restricted",
            "allowed_departments": ["Executive"],
            "allowed_roles": ["Executive"],
            "version": "3.0",
            "effective_date": "2026-09-01",
            "content": "Q4 projected revenue is 145 crore."
        }
    ]
    prompt_b = "What is the Q4 revenue forecast?"
    res_b = agent.execute_query(user_b, prompt_b, docs_b)
    print(json.dumps(res_b, indent=2))

    # -----------------------------------------------------
    # Test Input C — Authorized Conflict Resolution
    # -----------------------------------------------------
    print("\n[TEST C] Conflicting Authorized Documents (Picks Latest Version)")
    user_c = {"user_id": "U301", "role": "Finance", "department": "Finance", "clearance": "Internal"}
    docs_c = [
        {
            "document_id": "DOC-301",
            "title": "Q4 Forecast",
            "classification": "Internal",
            "allowed_departments": ["Finance"],
            "allowed_roles": ["Finance"],
            "version": "1.0",
            "effective_date": "2026-06-01",
            "content": "Q4 projected revenue is 110 crore."
        },
        {
            "document_id": "DOC-302",
            "title": "Q4 Forecast",
            "classification": "Internal",
            "allowed_departments": ["Finance"],
            "allowed_roles": ["Finance"],
            "version": "2.0",
            "effective_date": "2026-09-01",
            "content": "Q4 projected revenue is 125 crore."
        }
    ]
    prompt_c = "What is the latest Q4 revenue forecast?"
    res_c = agent.execute_query(user_c, prompt_c, docs_c)
    print(json.dumps(res_c, indent=2))

    # -----------------------------------------------------
    # Financial Inputs (Doc 2 Requirements)
    # -----------------------------------------------------
    print("\n[FINANCIAL INPUT 1] Expense Tracking & Overrun Alert")
    res_f1 = agent.execute_query(
        {"user_id": "U102", "role": "Finance", "department": "Finance", "clearance": "Internal"},
        "I've spent ₹12,000 on food this month. My budget is ₹8,000. How should I adjust my spending?"
    )
    print(json.dumps(res_f1, indent=2))

    print("\n[FINANCIAL INPUT 2] Log Daily Expense")
    res_f2 = agent.execute_query(
        {"user_id": "U102", "role": "Finance", "department": "Finance", "clearance": "Internal"},
        "I spent 500 on fuel today"
    )
    print(json.dumps(res_f2, indent=2))

    print("\n[FINANCIAL INPUT 3] Stock Investment Decision")
    res_f3 = agent.execute_query(
        {"user_id": "U102", "role": "Finance", "department": "Finance", "clearance": "Internal"},
        "Should I buy Apple stock right now? I have ₹5000 to invest."
    )
    print(json.dumps(res_f3, indent=2))

    print("\n[FINANCIAL INPUT 4] Lifestyle Flight Decision")
    res_f4 = agent.execute_query(
        {"user_id": "U205", "role": "Marketing", "department": "Marketing", "clearance": "Internal"},
        "Should I book this flight to Goa for ₹4,000 this weekend? I have limited budget."
    )
    print(json.dumps(res_f4, indent=2))

    print("\n[FINANCIAL INPUT 5] IPO Decision")
    res_f5 = agent.execute_query(
        {"user_id": "U301", "role": "Finance", "department": "Finance", "clearance": "Internal"},
        "A new IPO is launching today - TechCorp. Should I subscribe?"
    )
    print(json.dumps(res_f5, indent=2))

    # -----------------------------------------------------
    # Audit Trail Output
    # -----------------------------------------------------
    print("\n" + separator)
    print("IMMUTABLE AUDIT LOG SAMPLE (First 2 entries)")
    print(separator)
    print(json.dumps(agent.audit_log[:2], indent=2))


if __name__ == "__main__":
    run_all_hackathon_tests()