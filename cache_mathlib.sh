#!/bin/bash
# Mathlib Cache Optimization Script for Local Development
# This script helps manage and optimize mathlib caching for faster builds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Yang-Mills Proof - Mathlib Cache Manager"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get community mathlib cache (fastest setup)
get_community_cache() {
    print_status "🌐 Getting community mathlib cache..."
    
    if command -v lake &> /dev/null; then
        print_status "Using lake exe cache get for fastest setup..."
        if lake exe cache get; then
            print_success "✅ Community mathlib cache downloaded successfully!"
            print_status "📊 Cache size: $(du -sh .lake/packages/mathlib/.lake/build/ 2>/dev/null | cut -f1 || echo 'N/A')"
            return 0
        else
            print_warning "⚠️  Community cache download failed, falling back to local build..."
            return 1
        fi
    else
        print_error "Lake not found. Please install Lean 4 first."
        return 1
    fi
}

# Function to put cache back to community (for contributors)
put_community_cache() {
    print_status "🌐 Uploading to community mathlib cache..."
    
    if command -v lake &> /dev/null; then
        print_status "Using lake exe cache put to share cache..."
        if lake exe cache put; then
            print_success "✅ Cache uploaded to community successfully!"
            return 0
        else
            print_warning "⚠️  Cache upload failed - you may need proper permissions"
            return 1
        fi
    else
        print_error "Lake not found. Please install Lean 4 first."
        return 1
    fi
}

# Check current cache status
check_cache_status() {
    print_status "Checking current cache status..."
    
    if [ -d ".lake/packages/mathlib/.lake/build" ]; then
        MATHLIB_SIZE=$(du -sh .lake/packages/mathlib/.lake/build | cut -f1)
        print_success "Mathlib cache found: $MATHLIB_SIZE"
    else
        print_warning "No mathlib cache found"
        return 1
    fi
    
    if [ -d ".lake/build" ]; then
        PROJECT_SIZE=$(du -sh .lake/build | cut -f1)
        print_success "Project cache found: $PROJECT_SIZE"
    else
        print_warning "No project cache found"
    fi
    
    TOTAL_SIZE=$(du -sh .lake 2>/dev/null | cut -f1 || echo "0B")
    print_status "Total cache size: $TOTAL_SIZE"
    
    return 0
}

# Build mathlib dependencies with caching
build_mathlib_cached() {
    print_status "Building mathlib dependencies with caching..."
    
    # Create cache directory if it doesn't exist
    mkdir -p .lake/packages/mathlib/.lake/build
    
    # Build only the mathlib modules we need
    print_status "Building core mathlib modules..."
    lake build Mathlib.Data.Real.Basic || print_warning "Basic real numbers failed"
    lake build Mathlib.Data.Finset.Basic || print_warning "Finset basic failed"
    lake build Mathlib.Topology.Basic || print_warning "Topology basic failed"
    lake build Mathlib.Analysis.SpecialFunctions.Pow.Real || print_warning "Power functions failed"
    lake build Mathlib.Topology.Algebra.InfiniteSum.Basic || print_warning "Infinite sums failed"
    
    print_success "Mathlib core modules built"
}

# Clean cache selectively
clean_cache() {
    print_status "Cleaning cache selectively..."
    
    # Remove only project build, keep mathlib
    if [ -d ".lake/build" ]; then
        rm -rf .lake/build
        print_success "Project cache cleaned"
    fi
    
    # Keep mathlib cache intact
    print_status "Mathlib cache preserved"
}

# Full cache rebuild
rebuild_cache() {
    print_status "Rebuilding cache completely..."
    
    # Remove all cache
    rm -rf .lake/build
    rm -rf .lake/packages/*/lake/build
    
    # Update dependencies
    lake update
    
    # Rebuild mathlib
    build_mathlib_cached
    
    print_success "Cache rebuilt successfully"
}

# Backup cache
backup_cache() {
    BACKUP_DIR="cache_backup_$(date +%Y%m%d_%H%M%S)"
    print_status "Backing up cache to $BACKUP_DIR..."
    
    mkdir -p "$BACKUP_DIR"
    
    if [ -d ".lake/packages/mathlib/.lake/build" ]; then
        cp -r .lake/packages/mathlib/.lake/build "$BACKUP_DIR/mathlib_build"
        print_success "Mathlib cache backed up"
    fi
    
    if [ -d ".lake/build" ]; then
        cp -r .lake/build "$BACKUP_DIR/project_build"
        print_success "Project cache backed up"
    fi
    
    print_success "Cache backup completed: $BACKUP_DIR"
}

# Restore cache
restore_cache() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 restore <backup_directory>"
        return 1
    fi
    
    BACKUP_DIR="$1"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        print_error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    print_status "Restoring cache from $BACKUP_DIR..."
    
    if [ -d "$BACKUP_DIR/mathlib_build" ]; then
        mkdir -p .lake/packages/mathlib/.lake/
        cp -r "$BACKUP_DIR/mathlib_build" .lake/packages/mathlib/.lake/build
        print_success "Mathlib cache restored"
    fi
    
    if [ -d "$BACKUP_DIR/project_build" ]; then
        cp -r "$BACKUP_DIR/project_build" .lake/build
        print_success "Project cache restored"
    fi
    
    print_success "Cache restoration completed"
}

# Optimize cache for development
optimize_cache() {
    print_status "Optimizing cache for development..."
    
    # Remove macOS resource fork files that can cause issues
    find .lake -name "._*" -delete 2>/dev/null || true
    
    # Compress old .olean files
    find .lake -name "*.olean" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    
    # Clean up temporary files
    find .lake -name "*.tmp" -delete 2>/dev/null || true
    find .lake -name "*.log" -delete 2>/dev/null || true
    
    print_success "Cache optimized for development"
}

# Main command handling
case "${1:-status}" in
    "status")
        check_cache_status
        ;;
    "get")
        get_community_cache
        ;;
    "put")
        put_community_cache
        ;;
    "build")
        build_mathlib_cached
        ;;
    "clean")
        clean_cache
        ;;
    "rebuild")
        rebuild_cache
        ;;
    "backup")
        backup_cache
        ;;
    "restore")
        restore_cache "$2"
        ;;
    "optimize")
        optimize_cache
        ;;
    "help"|"--help"|"-h")
        echo "Usage: $0 [command]"
        echo ""
        echo "🌐 Community Cache Commands (Fastest):"
        echo "  get       - Download community mathlib cache (fastest first-time setup)"
        echo "  put       - Upload cache to community (for contributors)"
        echo ""
        echo "🔧 Local Cache Commands:"
        echo "  status    - Check current cache status (default)"
        echo "  build     - Build mathlib dependencies with caching"
        echo "  clean     - Clean project cache (keep mathlib)"
        echo "  rebuild   - Rebuild entire cache"
        echo "  backup    - Backup current cache"
        echo "  restore   - Restore cache from backup"
        echo "  optimize  - Optimize cache for development"
        echo "  help      - Show this help message"
        echo ""
        echo "💡 Quick Start: Use './cache_mathlib.sh get' for fastest setup"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 