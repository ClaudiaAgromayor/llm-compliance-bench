class LuggageFeeCalculator:
    """
    Versión mejorada del calculador de tarifas basado en funnel
    """
    
    def __init__(self, policy_text):
        """Initialize with the luggage policy"""
        self.policy = policy_text
        self._parse_policy()
        
    def _parse_policy(self):
        """Parse the policy document to extract fee structures"""
        # Base allowances by travel class - directly from the policy
        self.class_allowances = {
            'Economy': {'cabin': 1, 'checked': 1, 'cabin_weight': 7, 'checked_weight': 23},
            'Business': {'cabin': 1, 'checked': 2, 'cabin_weight': 10, 'checked_weight': 32},
            'First': {'cabin': 2, 'checked': 3, 'cabin_weight': 15, 'checked_weight': 32}
        }
        
        # Age category modifiers
        self.age_modifiers = {
            'adult': {'multiplier': 1.0},
            'child': {'multiplier': 0.8, 'allowance_mod': 0},
            'infant': {'multiplier': 0.5, 'allowance_mod': -1}
        }
        
        # Fee structure from the policy
        self.fee_structure = {
            'Economy': {
                'extra_bag': 60,
                'overweight': 15,  # per kg
                'oversize': 50
            },
            'Business': {
                'extra_bag': 100,
                'overweight': 20,  # per kg 
                'oversize': 75
            },
            'First': {
                'extra_bag': 120,
                'overweight': 30,  # per kg
                'oversize': 100
            }
        }
        
        # Maximum dimensions for each storage type
        self.max_dimensions = {
            'cabin': {'length': 55, 'width': 35, 'height': 25},
            'checked': {'length': 158, 'width': 75, 'height': 56}
        }
        
        # Absolute maximum weights that can't be exceeded (critical fix)
        self.absolute_max_weight = {
            'cabin': 20,  # Max cabin weight
            'checked': 32  # Max checked weight
        }
        
        # Special items that must go to cargo
        self.special_cargo_items = ["bike", "bicycle", "surfboard", "musical instrument"]
    
    def categorize_client(self, travel_class):
        """Stage 1: Categorize client based on travel class"""
        if travel_class not in self.class_allowances:
            raise ValueError(f"Unknown travel class: {travel_class}")
        return travel_class
    
    def apply_age_rules(self, travel_class, age_category):
        """Stage 2: Apply age-specific rules"""
        if age_category not in self.age_modifiers:
            raise ValueError(f"Unknown age category: {age_category}")
            
        base_allowance = self.class_allowances[travel_class].copy()
        modifier = self.age_modifiers[age_category]
        
        # Apply age-specific modifications
        if 'allowance_mod' in modifier:
            base_allowance['checked'] = max(0, base_allowance['checked'] + modifier['allowance_mod'])
            
        return base_allowance, modifier['multiplier']
    
    def calculate_luggage_fees(self, luggage_items, allowance, fee_structure, fee_multiplier=1.0):
        """Stage 3: Calculate specific fees based on luggage types - MEJORADO"""
        total_fees = 0
        cargo_items = []
        cabin_count = 0
        checked_count = 0
        compliance_issues = []
        
        # First pass: count items by storage type
        for item in luggage_items:
            storage = item.get('storage', '').lower()
            
            if storage == 'cabin':
                cabin_count += 1
            elif storage == 'checked':
                checked_count += 1
                
        # Second pass: process each luggage item in detail
        for item in luggage_items:
            storage = item.get('storage', '').lower()
            weight = float(item.get('weight', 0))
            height = float(item.get('height', 0))
            width = float(item.get('width', 0))
            depth = float(item.get('depth', 0))
            special = item.get('special', False)
            excess = item.get('excess', False)
            
            # Skip non-standard storage types
            if storage not in ['cabin', 'checked']:
                continue
            
            # *** CRITICAL FIX 1: Check for special items first ***
            if special:
                cargo_items.append(item)
                compliance_issues.append(f"Special item must go as cargo: {storage}")
                continue
                
            # *** CRITICAL FIX 2: Check excess flag ***
            if excess:
                if storage == 'cabin':
                    cargo_items.append(item)
                    compliance_issues.append(f"Excess cabin item must go as cargo")
                    continue
                else:
                    # Apply extra bag fee
                    total_fees += fee_structure['extra_bag'] * fee_multiplier
                    compliance_issues.append("Extra checked bag fee for excess item")
            
            # Check dimensions - fixed calculation
            is_oversize = False
            
            # *** CRITICAL FIX 3: Proper dimension calculation ***
            if (storage == 'cabin' and (
                    height > self.max_dimensions[storage]['height'] or
                    width > self.max_dimensions[storage]['width'] or
                    depth > self.max_dimensions[storage]['length'])):
                is_oversize = True
                # Cabin items that are oversize go to cargo
                cargo_items.append(item)
                compliance_issues.append("Oversized cabin item")
                continue
                
            elif (storage == 'checked' and (
                    height > self.max_dimensions[storage]['height'] or
                    width > self.max_dimensions[storage]['width'] or
                    depth > self.max_dimensions[storage]['length'])):
                is_oversize = True
                # Apply oversize fee for checked items
                total_fees += fee_structure['oversize'] * fee_multiplier
                compliance_issues.append("Oversized checked bag fee")
            
            # *** CRITICAL FIX 4: Weight limit check ***
            # Check absolute maximum weight first
            if weight > self.absolute_max_weight[storage]:
                cargo_items.append(item)
                compliance_issues.append(f"Item exceeds maximum weight for {storage}")
                continue
            
            # Check allowance count (quantity)
            if storage == 'cabin' and cabin_count > allowance['cabin']:
                # Too many cabin bags
                if checked_count < allowance['checked']:
                    # Convert to checked if space available
                    checked_count += 1
                    cabin_count -= 1
                else:
                    # Apply extra bag fee
                    total_fees += fee_structure['extra_bag'] * fee_multiplier
                    compliance_issues.append("Extra cabin bag fee applied")
            
            elif storage == 'checked' and checked_count > allowance['checked']:
                # Too many checked bags
                total_fees += fee_structure['extra_bag'] * fee_multiplier
                compliance_issues.append("Extra checked bag fee applied")
            
            # Check weight allowance and apply fees
            if storage == 'cabin' and weight > allowance['cabin_weight']:
                overweight = weight - allowance['cabin_weight']
                # For cabin bags, overweight > 5kg sends to cargo
                if overweight > 5:
                    cargo_items.append(item)
                    compliance_issues.append(f"Cabin bag over weight limit by {overweight}kg")
                else:
                    # Apply overweight fee
                    total_fees += overweight * fee_structure['overweight'] * fee_multiplier
                    compliance_issues.append(f"Cabin bag weight fee: {overweight}kg over")
            
            elif storage == 'checked' and weight > allowance['checked_weight']:
                overweight = weight - allowance['checked_weight']
                total_fees += overweight * fee_structure['overweight'] * fee_multiplier
                compliance_issues.append(f"Checked bag weight fee: {overweight}kg over")
            
        # *** CRITICAL FIX 5: Proper compliance logic ***
        # An item is non-compliant if:
        # 1. It has cargo items OR
        # 2. Total fees > 0
        is_compliant = len(cargo_items) == 0 and len(compliance_issues) == 0
                
        return {
            'total_fees': round(total_fees, 2),
            'cargo_items': cargo_items,
            'compliance_issues': compliance_issues,
            'is_compliant': is_compliant  # FIXED
        }
    
    def process_luggage(self, travel_class, age_category, luggage_items):
        """Execute the complete funnel process"""
        # Stage 1: Categorize by travel class
        category = self.categorize_client(travel_class)
        
        # Stage 2: Apply age rules
        allowance, fee_multiplier = self.apply_age_rules(category, age_category)
        
        # Stage 3: Calculate luggage fees
        fee_structure = self.fee_structure[category]
        result = self.calculate_luggage_fees(luggage_items, allowance, fee_structure, fee_multiplier)
        
        # Prepare final result
        compliance_message = "Luggage is compliant with policy" if result['is_compliant'] else "; ".join(result['compliance_issues'])
        
        return {
            "compliance_result": result['is_compliant'],
            "compliance_message": compliance_message,
            "cargo_items": [self._format_cargo_item(item) for item in result['cargo_items']],
            "fees": result['total_fees']
        }
    
    def _format_cargo_item(self, item):
        """Format a cargo item for output"""
        if isinstance(item, dict):
            desc = f"{item.get('storage', 'unknown')} bag"
            if 'weight' in item:
                desc += f" ({item['weight']}kg)"
            return desc
        return str(item)