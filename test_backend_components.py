#!/usr/bin/env python3
"""
Test if we can access Langflow components from the backend
"""

import sys
sys.path.insert(0, r'd:\langflow_spark\langflow-AI\src\backend\base')

try:
    from langflow.interface.types import get_type_dict
    print("✅ Successfully imported get_type_dict")
    
    all_types = get_type_dict()
    print(f"\n📦 Total component categories: {len(all_types)}")
    
    total_components = sum(len(comps) for comps in all_types.values() if isinstance(comps, dict))
    print(f"📦 Total components: {total_components}")
    
    print(f"\n🔍 Sample categories:")
    for category in list(all_types.keys())[:10]:
        components = all_types[category]
        if isinstance(components, dict):
            print(f"\n  {category} ({len(components)} components):")
            for comp_name in list(components.keys())[:5]:
                comp_data = components[comp_name]
                if isinstance(comp_data, dict):
                    display_name = comp_data.get('display_name', comp_name)
                    print(f"    - {comp_name} ({display_name})")
            if len(components) > 5:
                print(f"    ... and {len(components) - 5} more")
    
    # Test finding specific components
    print(f"\n🔍 Looking for Amazon components...")
    for category, components in all_types.items():
        if isinstance(components, dict):
            for comp_name in components.keys():
                if 'amazon' in comp_name.lower() or 's3' in comp_name.lower():
                    comp_data = components[comp_name]
                    display_name = comp_data.get('display_name', comp_name) if isinstance(comp_data, dict) else comp_name
                    print(f"  Found: {comp_name} ({display_name}) in category '{category}'")
    
except ImportError as e:
    print(f"❌ Cannot import get_type_dict: {e}")
    print("This means we need to ensure the backend path is correct")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
