#!/usr/bin/env python3
"""Test Marker OCR on Hebrew PDF"""

from marker.convert import convert_single_pdf
import sys

def test_marker():
    pdf_path = 'watched-dir/מכבי.pdf'
    print(f'Testing Marker on {pdf_path}...\n')
    
    try:
        # Convert PDF
        print('Converting PDF with Marker...')
        output = convert_single_pdf(pdf_path)
        
        # Extract text
        if hasattr(output, 'markdown'):
            markdown_text = output.markdown
        elif isinstance(output, tuple):
            markdown_text = output[0]
        else:
            markdown_text = str(output)
        
        print(f'\n=== MARKER OUTPUT ({len(markdown_text)} chars) ===')
        print(markdown_text[:2000])
        print('\n...\n')
        
        # Check for address fields
        print('=== ADDRESS CHECK ===')
        
        street_name = "ג'מילי"
        city_name = "פתח תקווה"
        phone_num = "0548859596"
        user_name = "תומר"
        
        if street_name in markdown_text or 'גמילי' in markdown_text:
            print(f'✓ Street name found: {street_name}')
        else:
            print('⚠️  Street name not found')
            
        if city_name in markdown_text or 'פתח-תקווה' in markdown_text:
            print(f'✓ City found: {city_name}')
        else:
            print('⚠️  City not found')
            
        if phone_num in markdown_text:
            print(f'✓ Phone found: {phone_num}')
        else:
            print('⚠️  Phone not found')
        
        if user_name in markdown_text:
            print(f'✓ Name found: {user_name}')
        else:
            print('⚠️  Name not found')
            
        # Check if רחוב and ג'מילי are on the same line (critical test)
        print('\n=== FIELD-VALUE STRUCTURE CHECK ===')
        lines = markdown_text.split('\n')
        
        street_found = False
        for i, line in enumerate(lines):
            if 'רחוב' in line:
                street_found = True
                print(f'Line {i}: {line}')
                if i+1 < len(lines):
                    print(f'Line {i+1}: {lines[i+1]}')
                    
                # Check if street name is on same line or next line
                if street_name in line or 'גמילי' in line:
                    print('✓✓✓ STREET NAME ON SAME LINE AS LABEL - GOOD!')
                elif i+1 < len(lines) and (street_name in lines[i+1] or 'גמילי' in lines[i+1]):
                    print('⚠️  Street name on next line - same issue as pytesseract')
                    
        if not street_found:
            print('⚠️  רחוב label not found in output')
            
        return True
        
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_marker()
    sys.exit(0 if success else 1)
