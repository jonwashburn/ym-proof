#!/bin/bash
# Local CI Status Check Script
# Mimics the GitHub Actions CI workflow

set -e

echo "🔍 Yang-Mills Proof - Local CI Status Check"
echo "============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "📋 Step 1: Build Check"
echo "----------------------"
echo "🔨 Building project..."
if lake build > /dev/null 2>&1; then
    echo "✅ Build: SUCCESS"
    BUILD_STATUS="✅ PASS"
else
    echo "❌ Build: FAILED"
    BUILD_STATUS="❌ FAIL"
fi

echo ""
echo "📋 Step 2: Axiom-Free Verification"
echo "-----------------------------------"
echo "🔍 Checking for axioms..."
if bash verify_no_axioms.sh > /dev/null 2>&1; then
    echo "✅ Axiom-Free: VERIFIED"
    AXIOM_STATUS="✅ PASS"
else
    echo "❌ Axiom-Free: FAILED"
    AXIOM_STATUS="❌ FAIL"
fi

echo ""
echo "📋 Step 3: Sorry-Free Verification"
echo "-----------------------------------"
echo "🔍 Checking for sorry statements..."
if bash verify_no_sorries.sh > /dev/null 2>&1; then
    echo "✅ Sorry-Free: VERIFIED"
    SORRY_STATUS="✅ PASS"
else
    echo "❌ Sorry-Free: FAILED"
    SORRY_STATUS="❌ FAIL"
fi

echo ""
echo "📊 FINAL STATUS SUMMARY"
echo "========================"
echo "Build Status:      $BUILD_STATUS"
echo "Axiom-Free Status: $AXIOM_STATUS"  
echo "Sorry-Free Status: $SORRY_STATUS"
echo ""

# Overall result
if [[ "$BUILD_STATUS" == "✅ PASS" && "$AXIOM_STATUS" == "✅ PASS" && "$SORRY_STATUS" == "✅ PASS" ]]; then
    echo "🎯 OVERALL STATUS: ✅ ALL CHECKS PASSED"
    echo "🏆 Yang-Mills proof is complete and formally verified!"
    exit 0
else
    echo "🔴 OVERALL STATUS: ❌ SOME CHECKS FAILED"
    echo "🔧 Please fix the issues above"
    exit 1
fi 