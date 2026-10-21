#!/bin/bash

# ============================================================================
# LaTeX Compilation Script for Beamer Presentation
# ============================================================================

# Configuration
FILENAME="presentation"
OUTPUT_DIR="output"
LOG_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
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

# Create directories if they don't exist
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
}

# Check if required files exist
check_files() {
    print_info "Checking for required files..."
    
    if [ ! -f "${FILENAME}.tex" ]; then
        print_error "Main file ${FILENAME}.tex not found!"
        exit 1
    fi
    
    if [ ! -d "img" ]; then
        print_warning "img directory not found. Images may not display correctly."
    fi
    
    if [ ! -f "ujlogo.pdf" ]; then
        print_warning "ujlogo.pdf not found. University logo may not display."
    fi
    
    if [ ! -f "ujslide.sty" ]; then
        print_warning "ujslide.sty not found. UJ theme may not work properly."
    fi
    
    print_success "File checks completed"
}

# Clean old files
clean_files() {
    print_info "Cleaning old auxiliary files..."
    
    # Remove LaTeX auxiliary files
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.bbl *.blg *.synctex.gz
    
    # Remove files in output directory
    rm -f "$OUTPUT_DIR"/*.pdf
    
    print_success "Cleaning completed"
}

# Compile LaTeX document
compile_latex() {
    print_info "Starting LaTeX compilation..."
    
    # First compilation pass
    print_info "Pass 1/2: Compiling..."
    pdflatex -interaction=nonstopmode -file-line-error "${FILENAME}.tex" > "${LOG_DIR}/compile_pass1.log" 2>&1
    
    if [ $? -ne 0 ]; then
        print_error "First compilation pass failed. Check ${LOG_DIR}/compile_pass1.log"
        grep -i "error" "${LOG_DIR}/compile_pass1.log" | head -10
        return 1
    fi
    
    # Check if we need a second pass (if references exist)
    if [ -f "${FILENAME}.aux" ] && grep -q "citation\|bibcite\|newlabel" "${FILENAME}.aux"; then
        print_info "References found. Running second pass..."
        
        # Second compilation pass for references
        pdflatex -interaction=nonstopmode -file-line-error "${FILENAME}.tex" > "${LOG_DIR}/compile_pass2.log" 2>&1
        
        if [ $? -ne 0 ]; then
            print_error "Second compilation pass failed. Check ${LOG_DIR}/compile_pass2.log"
            return 1
        fi
    fi
    
    print_success "LaTeX compilation completed successfully"
}

# Check if PDF was created successfully
check_pdf() {
    if [ -f "${FILENAME}.pdf" ]; then
        # Get PDF file size
        FILESIZE=$(du -h "${FILENAME}.pdf" | cut -f1)
        print_success "PDF generated successfully: ${FILENAME}.pdf ($FILESIZE)"
        
        # Move to output directory if enabled
        if [ "$1" = "--move" ] || [ "$1" = "-m" ]; then
            cp "${FILENAME}.pdf" "$OUTPUT_DIR/"
            print_info "Copied PDF to ${OUTPUT_DIR}/"
        fi
        
        return 0
    else
        print_error "PDF file not created!"
        return 1
    fi
}

# Display compilation statistics
show_stats() {
    echo ""
    echo "=========================================="
    echo "Compilation Statistics"
    echo "=========================================="
    
    # Count pages if PDF exists
    if [ -f "${FILENAME}.pdf" ]; then
        PAGES=$(pdfinfo "${FILENAME}.pdf" 2>/dev/null | grep "Pages" | awk '{print $2}')
        if [ -n "$PAGES" ]; then
            print_info "Number of pages: $PAGES"
        fi
    fi
    
    # Count warnings
    if [ -f "${FILENAME}.log" ]; then
        WARNINGS=$(grep -c "Warning" "${FILENAME}.log" 2>/dev/null || echo "0")
        print_warning "Number of warnings: $WARNINGS"
        
        # Show undefined references
        UNDEF_REFS=$(grep -c "undefined" "${FILENAME}.log" 2>/dev/null || echo "0")
        if [ "$UNDEF_REFS" -gt 0 ]; then
            print_warning "Undefined references: $UNDEF_REFS"
        fi
    fi
    
    echo "=========================================="
}

# Open PDF after compilation
open_pdf() {
    if [ -f "${FILENAME}.pdf" ]; then
        print_info "Opening PDF..."
        
        # Detect OS and use appropriate command
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            open "${FILENAME}.pdf"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v xdg-open > /dev/null; then
                xdg-open "${FILENAME}.pdf"
            elif command -v evince > /dev/null; then
                evince "${FILENAME}.pdf" &
            elif command -v okular > /dev/null; then
                okular "${FILENAME}.pdf" &
            else
                print_warning "No PDF viewer found. Please open ${FILENAME}.pdf manually."
            fi
        elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
            # Windows (Git Bash/Cygwin)
            start "${FILENAME}.pdf"
        fi
    fi
}

# Show usage
show_usage() {
    echo "Usage: ./compile.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --clean     Clean auxiliary files before compilation"
    echo "  -m, --move      Move PDF to output directory after compilation"
    echo "  -o, --open      Open PDF after successful compilation"
    echo "  -v, --verbose   Show verbose output"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./compile.sh                # Standard compilation"
    echo "  ./compile.sh -c             # Clean and compile"
    echo "  ./compile.sh -c -o          # Clean, compile, and open PDF"
    echo "  ./compile.sh -m -o          # Compile, move PDF, and open"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "LaTeX Compilation Script"
    echo "=========================================="
    echo ""
    
    # Parse arguments
    CLEAN_MODE=false
    MOVE_PDF=false
    OPEN_PDF=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--clean)
                CLEAN_MODE=true
                shift
                ;;
            -m|--move)
                MOVE_PDF=true
                shift
                ;;
            -o|--open)
                OPEN_PDF=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Setup directories
    setup_directories
    
    # Clean if requested
    if [ "$CLEAN_MODE" = true ]; then
        clean_files
    fi
    
    # Check required files
    check_files
    
    # Compile
    if compile_latex; then
        # Move PDF if requested
        if [ "$MOVE_PDF" = true ]; then
            cp "${FILENAME}.pdf" "$OUTPUT_DIR/"
            print_success "PDF copied to ${OUTPUT_DIR}/${FILENAME}.pdf"
        fi
        
        # Show statistics
        show_stats
        
        # Open PDF if requested
        if [ "$OPEN_PDF" = true ]; then
            open_pdf
        fi
        
        echo ""
        print_success "Done! 🎉"
        echo ""
    else
        echo ""
        print_error "Compilation failed! Please check the log files in ${LOG_DIR}/"
        echo ""
        exit 1
    fi
}

# Run main function with all arguments
main "$@"