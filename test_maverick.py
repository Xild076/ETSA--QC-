try:
    from maverick import Maverick
    print('Maverick is available')
except ImportError as e:
    print(f'Maverick not available: {e}')
