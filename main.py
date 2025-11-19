"""
Main entry point for the Plagiarism Checking System with CrewAI
"""
from app import PlagiarismCheckSystem
import json


def main():
    """Main function to run the plagiarism checking system"""
    
    # Initialize the system
    plagiarism_system = PlagiarismCheckSystem()
    
    print("\n" + "="*70)
    print("PLAGIARISM DETECTION SYSTEM WITH CREWAI")
    print("="*70)
    
    # Example programming question
    question = """Write a Python function that finds the sum of all prime numbers less than a given number n.
    The function should be efficient and handle edge cases."""
    
    # Example user submitted code
    user_code = """def sum_of_primes(n):
    if n <= 2:
        return 0
    
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    return sum(primes)"""
    
    # Run full analysis (both code generation and plagiarism detection)
    result = plagiarism_system.full_analysis(question, user_code)
    
    print("\n" + "="*70)
    print("FINAL ANALYSIS REPORT")
    print("="*70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
