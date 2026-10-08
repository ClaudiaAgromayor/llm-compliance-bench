import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from abstract_policy import Policy
from luggage import Luggage
from luggage_compliance_request import LuggageComplianceRequest

import unittest


class LuggageCompliance(Policy):
    def __init__(self):
        self.classes = {
            "Economy": {
                "carry_on": {"quantity": 1, "weight_limit": 7, "size_limit": [55, 40, 23]},
                "checked": {"allowance": 1, "weight_limit": 23, "size_limit": 158}
            },
            "Business": {
                "carry_on": {"quantity": 2, "weight_limit": 12, "size_limit": [55, 40, 23]},
                "checked": {"allowance": 2, "weight_limit": 32, "size_limit": 158}
            },
            "First": {
                "carry_on": {"quantity": 2, "weight_limit": 12, "size_limit": [55, 40, 23]},
                "checked": {"allowance": 3, "weight_limit": 32, "size_limit": 158}
            }
        }
        
        # Nouvelle structure pour les allocations spéciales
        self.infant_policy = {
            "carry_on": {"quantity": 1, "weight_limit": 10, "size_limit": [55, 40, 23]},
            "checked": {"allowance": 0},  # Pas de bagage enregistré
            "additional": "collapsible stroller"
        }
        
        self.child_allowances = {
            "child": {"checked": {"allowance": 1, "weight_limit": 23, "size_limit": 158}}
        }
        
        self.excess_fees = {
            "overweight": 75,
            "oversize": 100,
            "extra_piece": 150
        }
        
        
    def validate_carry_on(self, travel_class, carry_on_items, personal_items, passenger_type="adult"):
        # Déterminer la politique applicable
        if passenger_type == "infant":
            class_policy = self.infant_policy["carry_on"]
            # Les nourrissons n'ont pas droit aux personal items
            if len(personal_items) > 0:
                return False, "Infants are not allowed personal items, only one carry-on bag."
        else:
            class_policy = self.classes[travel_class]["carry_on"]
        
        # Vérifier d'abord si un bagage individuel est trop lourd (sécurité)
        for item in carry_on_items + personal_items:
            if item.weight > 32:  # Limite de sécurité absolue
                return False, f"Carry-on/personal item weighing {item.weight}kg exceeds absolute safety limit of 32kg. This item must be checked."
        
        # Compute total weight of carry-on and personal items
        total_weight = sum(item.weight for item in carry_on_items + personal_items)
        
        # Check quantity
        expected_quantity = class_policy["quantity"] + (0 if passenger_type == "infant" else 1)
        if len(carry_on_items) + len(personal_items) > expected_quantity:
            return False, "Exceeded carry-on quantity allowance (including personal items)."
        
        # Check weight
        if total_weight > class_policy["weight_limit"]:
            return False, f"Exceeded carry-on weight limit ({class_policy['weight_limit']}kg)."
        
        # Check size
        for item in carry_on_items:
            if any(dim > lim for dim, lim in zip([item.dim[k]
                                                for k in ["height", "width", "depth"]],
                                                class_policy["size_limit"])):
                return False, "Carry-on bag exceeds size limits."
        
        return True, "Carry-on luggage complies with the policy."

    def validate_checked_baggage(self, travel_class, checked_items, passenger_type="adult"):
        # Copie pour éviter de modifier l'original
        class_policy = self.classes[travel_class]["checked"].copy()
        fees = 0
        cargo_items = []
        
        # Gestion spéciale pour enfants et nourrissons
        if passenger_type == "child":
            # Les enfants ont la même franchise que les adultes
            pass  # Pas de modification nécessaire
        elif passenger_type == "infant":
            # Les nourrissons n'ont PAS de bagage enregistré
            class_policy["allowance"] = 0
            
        for item in checked_items:
            weight = item.weight
            # Calcul correct des dimensions totales
            total_size = item.dim["height"] + item.dim["width"] + item.dim["depth"]
            
            # Vérifier d'abord les limites de poids selon la classe/catégorie
            weight_limit = class_policy["weight_limit"]
            
            # Check si le bagage doit aller en cargo
            # Check si le bagage doit aller en cargo
            if weight > 32 or total_size > 203:
                cargo_items.append(item)
                # IMPORTANT: Les bagages cargo génèrent quand même des frais
                # Pour les enfants en Economy, la limite reste 23kg
                actual_weight_limit = weight_limit
                if passenger_type == "child" and travel_class == "Economy":
                    actual_weight_limit = 23  # Limite Economy standard
                
                if weight > actual_weight_limit and weight <= 32:
                    fees += self.excess_fees["overweight"]
                elif weight > 32:
                    # Pour les bagages > 32kg, calculer les frais sur la base de la limite de classe
                    fees += self.excess_fees["overweight"]
                
                if total_size > class_policy["size_limit"] and total_size <= 203:
                    fees += self.excess_fees["oversize"]
                continue        
            
            # Pour les bagages non-cargo, calculer les frais normalement
            if weight > weight_limit:
                fees += self.excess_fees["overweight"]
                
            if total_size > class_policy["size_limit"]:
                fees += self.excess_fees["oversize"]
        
        # Calculer les frais pour bagages supplémentaires
        # Ne pas compter les bagages cargo dans le calcul du nombre autorisé
        excess_items = max(0, len(checked_items) - class_policy["allowance"])
        fees += self.excess_fees["extra_piece"] * excess_items
        
        if cargo_items:
            cargo_reasons = []
            for item in cargo_items:
                weight = item.weight
                total_size = item.dim["height"] + item.dim["width"] + item.dim["depth"]
                reasons = []
                
                if weight > 32:
                    reasons.append(f"weight {weight}kg exceeds 32kg limit")
                if total_size > 203:
                    reasons.append(f"total dimensions {total_size:.1f}cm exceed 203cm limit")
                
                cargo_reasons.append(f"Bag {checked_items.index(item)+1}: {' and '.join(reasons)}")
            
            cargo_message = f"The following items must be shipped as cargo: {'; '.join(cargo_reasons)}. "
            cargo_message += f"Total fees: €{fees} (cargo shipping costs not included - please contact cargo department). "
            cargo_message += "Consider splitting items or reducing weight/size to avoid cargo shipping."
            
            return False, cargo_message, cargo_items, fees
        
        return True, "Checked luggage complies with the policy.", cargo_items, fees

    def test_eligibility(self, request):
        # Extraire les bagages par type
        carry_on_items = [x for x in request.luggages if x.storage == "carry-on"]
        personal_items = [x for x in request.luggages if x.storage == "personal"]
        checked_items = [x for x in request.luggages if x.storage == "checked"]
        
        # Valider les bagages cabine
        carry_on_result, carry_on_message = self.validate_carry_on(
        request.travel_class, carry_on_items, personal_items, request.age_category
        )
    
        
        # Valider les bagages enregistrés (toujours vérifier pour identifier les cargo)
        checked_result, checked_message, cargo_items, fees = self.validate_checked_baggage(
            request.travel_class, checked_items, request.age_category
        )
        
        # Déterminer la conformité globale
        # Conforme = pas de problème cabine + pas de cargo + pas de frais
        is_compliant = carry_on_result and checked_result and len(cargo_items) == 0 and fees == 0
        
        # Déterminer le message final
        # Déterminer le message final avec détails
        if not carry_on_result:
            final_message = carry_on_message
        elif cargo_items:
            final_message = checked_message  # Déjà détaillé avec la modification ci-dessus
        elif fees > 0:
            # Détailler les frais
            fee_details = []
            
            # Analyser pourquoi il y a des frais
            for item in checked_items:
                weight = item.weight
                total_size = item.dim["height"] + item.dim["width"] + item.dim["depth"]
                weight_limit = self.classes[request.travel_class]["checked"]["weight_limit"]
                
                if request.age_category == "infant":
                    weight_limit = 10
                
                if weight > weight_limit and weight <= 32:
                    fee_details.append(f"overweight baggage ({weight}kg > {weight_limit}kg limit): €75")
                if total_size > 158 and total_size <= 203:
                    fee_details.append(f"oversize baggage ({total_size:.1f}cm > 158cm limit): €100")
            
            # Vérifier les bagages supplémentaires
            allowance = self.classes[request.travel_class]["checked"]["allowance"]
            if request.age_category == "infant":
                allowance = 0
            
            excess_count = len(checked_items) - allowance
            if excess_count > 0:
                fee_details.append(f"{excess_count} extra bag(s) beyond {allowance} allowed: €{150 * excess_count}")
            
            final_message = f"Additional fees apply (total: €{fees}): {'; '.join(fee_details)}"
        else:
            final_message = "All luggage complies with the policy."
        
        return is_compliant, final_message, cargo_items, fees


def test1():
    # Instantiate the policy
    policy = LuggageCompliance()
    travel_class = "Economy"

    # Test case: Validate carry-on and checked luggage for Economy class
    bag1 = Luggage(storage="carry-on", weight=5.0, dim={"height": 50.0, "width": 40.0, "depth": 23.0, "unit": "cm"})
    bag2 = Luggage(storage="checked", weight=25.0, dim={"height": 40.0, "width": 30.0, "depth": 30.0, "unit": "cm"})
    bag3 = Luggage(storage="personal", weight=4.0, dim={"height": 20.0, "width": 50.0, "depth": 30.0, "unit": "cm"})
    compliance_request = LuggageComplianceRequest(travel_class="Business", age_category="adult",
                                                  luggages=[bag1, bag2, bag3])

    result = policy.test_eligibility(compliance_request)
    print(result)


class TestLuggageCompliance(unittest.TestCase):

    def setUp(self):
        self.policy = LuggageCompliance()

    def test_carry_on_exceeds_quantity(self):
        """Carry-on and personal items exceed quantity limit."""
        bag1 = Luggage(storage="carry-on", weight=5.0, dim={"height": 55, "width": 40, "depth": 20, "unit": "cm"})
        bag2 = Luggage(storage="carry-on", weight=5.0, dim={"height": 55, "width": 40, "depth": 23, "unit": "cm"})
        bag3 = Luggage(storage="personal", weight=2.0, dim={"height": 30, "width": 20, "depth": 10, "unit": "cm"})
        bag4 = Luggage(storage="personal", weight=2.0, dim={"height": 30, "width": 20, "depth": 10, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1, bag2, bag3, bag4])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertIn("Exceeded carry-on quantity allowance", result[1])

    def test_carry_on_exceeds_weight(self):
        """Combined carry-on weight exceeds the limit."""
        bag1 = Luggage(storage="carry-on", weight=8.0, dim={"height": 55, "width": 40, "depth": 20, "unit": "cm"})
        bag2 = Luggage(storage="personal", weight=3.0, dim={"height": 30, "width": 20, "depth": 10, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1, bag2])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertIn("Exceeded carry-on weight limit", result[1])

    def test_carry_on_exceeds_size(self):
        """Carry-on bag exceeds size limits."""
        bag1 = Luggage(storage="carry-on", weight=5.0, dim={"height": 60, "width": 45, "depth": 30, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertIn("Carry-on bag exceeds size limits", result[1])

    def test_checked_baggage_exceeds_allowance(self):
        """Checked baggage exceeds allowance."""
        bag1 = Luggage(storage="checked", weight=20.0, dim={"height": 70, "width": 50, "depth": 30, "unit": "cm"})
        bag2 = Luggage(storage="checked", weight=20.0, dim={"height": 70, "width": 50, "depth": 30, "unit": "cm"})
        bag3 = Luggage(storage="checked", weight=20.0, dim={"height": 70, "width": 50, "depth": 30, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1, bag2, bag3])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])  # Non-compliant car il y a des frais
        self.assertEqual(result[3], 300)  # 2 bagages supplémentaires × 150€

    def test_checked_baggage_overweight(self):
        """Checked baggage is overweight."""
        bag1 = Luggage(storage="checked", weight=33.0, dim={"height": 70, "width": 50, "depth": 30, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertIn("The following items must be shipped as cargo", result[1])
        self.assertIn("weight 33.0kg exceeds 32kg limit", result[1])

    def test_checked_baggage_oversized(self):
        """Checked baggage is oversized but within limits."""
        bag1 = Luggage(storage="checked", weight=25.0, dim={"height": 100, "width": 60, "depth": 50, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertGreater(result[3], 0)

    def test_checked_baggage_max_exceeded(self):
        """Checked baggage is too large or too heavy."""
        bag1 = Luggage(storage="checked", weight=35.0, dim={"height": 100, "width": 80, "depth": 50, "unit": "cm"})
        compliance_request = LuggageComplianceRequest("Economy", "adult", [bag1])

        result = self.policy.test_eligibility(compliance_request)
        self.assertFalse(result[0])
        self.assertIn("The following items must be shipped as cargo", result[1])
        self.assertIn("weight 35.0kg exceeds 32kg limit", result[1])
        self.assertIn("total dimensions 230.0cm exceed 203cm limit", result[1])
        self.assertGreater(result[3], 0)


if __name__ == "__main__":
    unittest.main()
