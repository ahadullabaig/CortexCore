/**
 * Accessibility Helper Functions
 * WCAG 2.1 Level A/AA compliance utilities
 */

/**
 * Set button enabled/disabled state with ARIA synchronization
 * CRITICAL: Fixes WCAG 2.1 Level A violation (4.1.2 Name, Role, Value)
 *
 * @param {HTMLButtonElement} button - Button element to update
 * @param {boolean} isDisabled - Whether button should be disabled
 */
export function setButtonState(button, isDisabled) {
    if (!button) {
        console.error('❌ setButtonState: button element is null');
        return;
    }

    button.disabled = isDisabled;
    button.setAttribute('aria-disabled', isDisabled.toString());

    // Log state change for debugging
    console.log(`🔘 Button "${button.id}" state: ${isDisabled ? 'disabled' : 'enabled'}`);
}

/**
 * Announce message to screen readers using ARIA live region
 * Phase 4: Accessibility enhancement
 * @param {string} message - Message to announce
 */
export function announceToScreenReader(message) {
    // Create or get announcement container
    let announcer = document.getElementById('aria-announcer');

    if (!announcer) {
        announcer = document.createElement('div');
        announcer.id = 'aria-announcer';
        announcer.className = 'sr-only'; // Screen reader only (visually hidden)
        announcer.setAttribute('role', 'status');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        document.body.appendChild(announcer);
    }

    // Clear previous announcement
    announcer.textContent = '';

    // Add new announcement after brief delay (ensures screen reader picks it up)
    setTimeout(() => {
        announcer.textContent = message;
    }, 100);
}
