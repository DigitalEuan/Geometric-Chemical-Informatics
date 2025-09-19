#!/usr/bin/env python3
"""
Materials Project API Field Explorer
Check available fields and update our acquisition script
"""

from mp_api.client import MPRester
import json

def explore_mp_fields():
    """Explore available fields in Materials Project API"""
    
    api_key = "QgtXNTeNADmxywaU4nAWo7oI5aT8J4g4"
    
    print("="*60)
    print("MATERIALS PROJECT API FIELD EXPLORATION")
    print("="*60)
    
    with MPRester(api_key) as mpr:
        
        # Check available search methods
        print("Available search methods:")
        methods = [method for method in dir(mpr) if not method.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        
        print("\\n" + "="*40)
        print("MATERIALS SEARCH FIELDS")
        print("="*40)
        
        # Try to get a small sample to see available fields
        try:
            # Get one document to see structure
            docs = mpr.materials.search(
                elements=["Fe"],
                num_elements=(1, 2),
                limit=1
            )
            
            if docs:
                doc = docs[0]
                print("Available fields in materials search:")
                for field in sorted(dir(doc)):
                    if not field.startswith('_'):
                        try:
                            value = getattr(doc, field)
                            print(f"  - {field}: {type(value).__name__}")
                        except:
                            print(f"  - {field}: (property)")
                
                print("\\nSample document structure:")
                print(json.dumps(doc.model_dump(), indent=2, default=str)[:1000] + "...")
        
        except Exception as e:
            print(f"Error exploring materials: {e}")
        
        # Check other endpoints
        print("\\n" + "="*40)
        print("OTHER AVAILABLE ENDPOINTS")
        print("="*40)
        
        endpoints = ['materials', 'thermo', 'electronic_structure', 'magnetism', 'phonon']
        for endpoint in endpoints:
            if hasattr(mpr, endpoint):
                print(f"✅ {endpoint}")
            else:
                print(f"❌ {endpoint}")
        
        # Try electronic structure endpoint
        if hasattr(mpr, 'electronic_structure'):
            print("\\nElectronic structure fields:")
            try:
                es_docs = mpr.electronic_structure.search(
                    material_ids=["mp-149"],  # Iron
                    limit=1
                )
                if es_docs:
                    es_doc = es_docs[0]
                    for field in sorted(dir(es_doc)):
                        if not field.startswith('_'):
                            print(f"  - {field}")
            except Exception as e:
                print(f"Error exploring electronic structure: {e}")
        
        # Try magnetism endpoint
        if hasattr(mpr, 'magnetism'):
            print("\\nMagnetism fields:")
            try:
                mag_docs = mpr.magnetism.search(
                    material_ids=["mp-149"],  # Iron
                    limit=1
                )
                if mag_docs:
                    mag_doc = mag_docs[0]
                    for field in sorted(dir(mag_doc)):
                        if not field.startswith('_'):
                            print(f"  - {field}")
            except Exception as e:
                print(f"Error exploring magnetism: {e}")

if __name__ == "__main__":
    explore_mp_fields()
