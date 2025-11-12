/**
 * PHASE 4: Dynamic Plotly Theme System
 *
 * Plotly Theme Configuration - Reads from CSS Variables
 * This creates a centralized theme that stays in sync with CSS
 */

export const PlotlyTheme = {
    // Cache for CSS variables (populated on first access)
    _cssVars: null,

    /**
     * Read CSS custom properties from document
     * Cached for performance
     */
    getCSSVariables() {
        if (!this._cssVars) {
            const root = document.documentElement;
            const computed = getComputedStyle(root);

            this._cssVars = {
                // Fonts
                fontPrimary: computed.getPropertyValue('--font-primary').trim() || 'JetBrains Mono, monospace',

                // Colors - Text
                textPrimary: computed.getPropertyValue('--text-primary').trim() || '#e4e7ed',
                textSecondary: computed.getPropertyValue('--text-secondary').trim() || '#b4bcd0',
                textTertiary: computed.getPropertyValue('--text-tertiary').trim() || '#8b92a0',

                // Colors - Accent
                neuralCyan: computed.getPropertyValue('--neural-cyan').trim() || '#00d9ff',
                neuralBlue: computed.getPropertyValue('--neural-blue').trim() || '#0088ff',
                neuralPurple: computed.getPropertyValue('--neural-purple').trim() || '#a855f7',
                neuralTeal: computed.getPropertyValue('--neural-teal').trim() || '#14b8a6',

                // Colors - Clinical
                clinicalNormal: computed.getPropertyValue('--clinical-normal').trim() || '#10b981',
                clinicalCritical: computed.getPropertyValue('--clinical-critical').trim() || '#ef4444',
                clinicalWarning: computed.getPropertyValue('--clinical-warning').trim() || '#f59e0b',
                clinicalInfo: computed.getPropertyValue('--clinical-info').trim() || '#3b82f6',

                // Backgrounds
                bgPrimary: computed.getPropertyValue('--bg-primary').trim() || '#0a0e1a',
                bgSecondary: computed.getPropertyValue('--bg-secondary').trim() || '#0f1419',

                // Borders
                borderSubtle: computed.getPropertyValue('--border-subtle').trim() || '#1f2937',
            };

            console.log('🎨 Plotly theme initialized with CSS variables:', this._cssVars);
        }

        return this._cssVars;
    },

    /**
     * Invalidate CSS variable cache
     * Call this if CSS theme changes dynamically
     */
    invalidateCache() {
        this._cssVars = null;
        console.log('🔄 Plotly theme cache invalidated');
    },

    /**
     * Generate base layout configuration
     * Applies to all Plotly charts for consistency
     */
    getBaseLayout(customTitle = '') {
        const vars = this.getCSSVariables();

        return {
            title: {
                text: customTitle,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textSecondary,
                    size: 16
                }
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent to show card background
            paper_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent
            margin: { t: 60, r: 20, b: 60, l: 70 },
            font: {
                family: vars.fontPrimary,
                color: vars.textSecondary,
                size: 12
            },
            hoverlabel: {
                bgcolor: vars.bgSecondary,
                bordercolor: vars.neuralCyan,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textPrimary,
                    size: 12
                }
            },
            // Modebar (toolbar) styling
            modebar: {
                bgcolor: 'rgba(0, 0, 0, 0)',
                color: vars.textTertiary,
                activecolor: vars.neuralCyan
            }
        };
    },

    /**
     * Generate axis configuration
     */
    getAxisConfig(title = '', options = {}) {
        const vars = this.getCSSVariables();

        return {
            title: {
                text: title,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textSecondary,
                    size: 13
                }
            },
            gridcolor: `rgba(180, 188, 208, 0.08)`, // Subtle grid
            zerolinecolor: `rgba(180, 188, 208, 0.15)`, // Slightly more visible zero line
            color: vars.textTertiary, // Tick labels
            tickfont: {
                family: vars.fontPrimary,
                size: 11
            },
            ...options // Allow overrides
        };
    },

    /**
     * Get color for condition/classification
     */
    getConditionColor(condition) {
        const vars = this.getCSSVariables();

        const colorMap = {
            'normal': vars.clinicalNormal,
            'arrhythmia': vars.clinicalCritical,
            'warning': vars.clinicalWarning,
            'info': vars.clinicalInfo
        };

        return colorMap[condition.toLowerCase()] || vars.neuralCyan;
    },

    /**
     * Get spike/neural activity color
     */
    getSpikeColor() {
        const vars = this.getCSSVariables();
        return vars.neuralPurple;
    },

    /**
     * Get Plotly configuration with accessible modebar
     * WCAG 2.1 AA: Ensures all chart controls have descriptive labels
     *
     * @param {Object} options - Override options
     * @returns {Object} Plotly config object
     */
    getAccessibleConfig(options = {}) {
        return {
            responsive: true,
            displayModeBar: true,
            displaylogo: false, // Remove Plotly logo
            // Remove confusing selection tools, keep essential zoom/pan controls
            modeBarButtonsToRemove: [
                'lasso2d',      // Lasso select (not useful for time series)
                'select2d'      // Box select (not useful for time series)
            ],
            // Enable proper ARIA labels via locale
            locale: 'en',
            // Ensure tooltips are shown (default true, but explicit for clarity)
            displayModeBarTooltips: true,
            ...options // Allow overrides
        };
    }
};

/**
 * Wait for fonts to load before initializing Plotly
 * Prevents FOUT (Flash of Unstyled Text) in charts
 */
export async function waitForFonts() {
    try {
        // Check if document.fonts API is available
        if ('fonts' in document) {
            await document.fonts.ready;
            console.log('✅ Fonts loaded, ready for Plotly rendering');
        } else {
            // Fallback: wait 100ms for fonts to load
            console.warn('⚠️ document.fonts API not available, using fallback delay');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    } catch (error) {
        console.warn('⚠️ Font loading check failed:', error);
    }
}
