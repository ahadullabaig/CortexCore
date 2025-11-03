"""
Code Quality & Security Review
===============================

Reviews code for:
- Best practices
- Potential bugs
- Security vulnerabilities
- Performance issues
- Edge cases
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any

class CodeReviewer:
    def __init__(self):
        self.issues = {
            'critical': [],
            'warnings': [],
            'suggestions': [],
            'best_practices': []
        }
        self.metrics = {
            'total_functions': 0,
            'functions_with_docstrings': 0,
            'functions_with_type_hints': 0,
            'total_classes': 0,
            'classes_with_docstrings': 0
        }

    def review_file(self, file_path: Path):
        """Review a single Python file"""
        print(f"\n📄 Reviewing: {file_path}")
        print("   " + "─" * 56)

        with open(file_path, 'r') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
            self.analyze_ast(tree, file_path)
        except SyntaxError as e:
            self.issues['critical'].append(f"{file_path}: Syntax error - {e}")
            return

        # Additional checks
        self.check_security_issues(content, file_path)
        self.check_error_handling(content, file_path)
        self.check_best_practices(content, file_path)

    def analyze_ast(self, tree, file_path):
        """Analyze AST for quality metrics"""
        for node in ast.walk(tree):
            # Functions
            if isinstance(node, ast.FunctionDef):
                self.metrics['total_functions'] += 1

                # Check for docstring
                if ast.get_docstring(node):
                    self.metrics['functions_with_docstrings'] += 1
                else:
                    self.issues['suggestions'].append(
                        f"{file_path}:{node.lineno} - Function '{node.name}' missing docstring"
                    )

                # Check for type hints
                if node.returns or any(arg.annotation for arg in node.args.args):
                    self.metrics['functions_with_type_hints'] += 1

            # Classes
            elif isinstance(node, ast.ClassDef):
                self.metrics['total_classes'] += 1

                if ast.get_docstring(node):
                    self.metrics['classes_with_docstrings'] += 1
                else:
                    self.issues['suggestions'].append(
                        f"{file_path}:{node.lineno} - Class '{node.name}' missing docstring"
                    )

    def check_security_issues(self, content: str, file_path: Path):
        """Check for security vulnerabilities"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for eval/exec usage
            if 'eval(' in line or 'exec(' in line:
                self.issues['critical'].append(
                    f"{file_path}:{i} - Unsafe use of eval/exec"
                )

            # Check for hardcoded credentials (basic check)
            if any(keyword in line.lower() for keyword in ['password =', 'api_key =', 'secret =']):
                if '=' in line and '"' in line or "'" in line:
                    self.issues['warnings'].append(
                        f"{file_path}:{i} - Possible hardcoded credential"
                    )

            # Check for SQL injection potential
            if 'execute(' in line and '%' in line:
                self.issues['warnings'].append(
                    f"{file_path}:{i} - Potential SQL injection risk (use parameterized queries)"
                )

    def check_error_handling(self, content: str, file_path: Path):
        """Check error handling practices"""
        lines = content.split('\n')

        in_except = False
        except_line = 0

        for i, line in enumerate(lines, 1):
            if 'except:' in line or 'except Exception:' in line:
                self.issues['warnings'].append(
                    f"{file_path}:{i} - Broad exception catching (be more specific)"
                )

            if 'except' in line and 'pass' in lines[min(i, len(lines)-1)]:
                self.issues['warnings'].append(
                    f"{file_path}:{i} - Silent exception handling (exceptions should be logged)"
                )

    def check_best_practices(self, content: str, file_path: Path):
        """Check for Python best practices"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for mutable default arguments
            if 'def ' in line and ('=[]' in line or '={}' in line):
                self.issues['warnings'].append(
                    f"{file_path}:{i} - Mutable default argument (use None instead)"
                )

            # Check for bare print statements (should use logging)
            if file_path.name != 'app.py' and 'print(' in line and not line.strip().startswith('#'):
                # Allow prints in scripts
                if 'scripts' not in str(file_path):
                    self.issues['suggestions'].append(
                        f"{file_path}:{i} - Use logging instead of print"
                    )

    def review_all_src_files(self):
        """Review all source files"""
        print("="*60)
        print("🔍 CODE QUALITY & SECURITY REVIEW")
        print("="*60)

        src_files = list(Path('src').glob('*.py'))
        src_files += list(Path('demo').glob('*.py'))

        for file_path in src_files:
            if file_path.name != '__init__.py':
                self.review_file(file_path)

    def print_summary(self):
        """Print review summary"""
        print("\n" + "="*60)
        print("📊 CODE REVIEW SUMMARY")
        print("="*60)

        # Metrics
        print("\n📈 Code Metrics:")
        print(f"   Functions: {self.metrics['total_functions']}")
        print(f"   Functions with docstrings: {self.metrics['functions_with_docstrings']}")
        print(f"   Docstring coverage: {(self.metrics['functions_with_docstrings'] / self.metrics['total_functions'] * 100):.1f}%")
        print(f"   Functions with type hints: {self.metrics['functions_with_type_hints']}")
        print(f"   Classes: {self.metrics['total_classes']}")
        print(f"   Classes with docstrings: {self.metrics['classes_with_docstrings']}")

        # Issues
        total_issues = (
            len(self.issues['critical']) +
            len(self.issues['warnings']) +
            len(self.issues['suggestions'])
        )

        print(f"\n🔍 Issues Found: {total_issues}")
        print(f"   ❌ Critical: {len(self.issues['critical'])}")
        print(f"   ⚠️  Warnings: {len(self.issues['warnings'])}")
        print(f"   💡 Suggestions: {len(self.issues['suggestions'])}")

        # Critical issues
        if self.issues['critical']:
            print("\n❌ CRITICAL ISSUES:")
            for issue in self.issues['critical'][:5]:  # Show first 5
                print(f"   - {issue}")

        # Warnings
        if self.issues['warnings']:
            print("\n⚠️  WARNINGS:")
            for issue in self.issues['warnings'][:5]:  # Show first 5
                print(f"   - {issue}")

        # Suggestions
        if self.issues['suggestions']:
            print("\n💡 SUGGESTIONS (first 5):")
            for issue in self.issues['suggestions'][:5]:
                print(f"   - {issue}")

        # Overall assessment
        print("\n" + "="*60)
        if len(self.issues['critical']) == 0:
            print("✅ CODE QUALITY: NO CRITICAL ISSUES FOUND")
            if len(self.issues['warnings']) == 0:
                print("✅ CODE QUALITY: EXCELLENT - NO WARNINGS")
            else:
                print(f"⚠️  CODE QUALITY: GOOD - {len(self.issues['warnings'])} warnings to address")
        else:
            print(f"❌ CODE QUALITY: {len(self.issues['critical'])} critical issues to fix")
        print("="*60)

if __name__ == "__main__":
    reviewer = CodeReviewer()
    reviewer.review_all_src_files()
    reviewer.print_summary()
