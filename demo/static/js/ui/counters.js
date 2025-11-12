/**
 * Number Counter Animations
 * Smooth count-up animations with easing
 */

/**
 * Animate count-up from start to end value
 * @param {HTMLElement} element - Element to animate
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} duration - Animation duration in ms
 * @param {Object} options - { suffix, decimals }
 */
export function animateCountUp(element, start, end, duration = 1000, options = {}) {
    const { suffix = '', decimals = 0 } = options;
    const startTime = performance.now();
    const range = end - start;

    function updateCount(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        const current = start + range * easeProgress;
        const formattedValue = decimals > 0 ? current.toFixed(decimals) : Math.floor(current).toLocaleString();
        element.textContent = formattedValue + suffix;

        if (progress < 1) {
            requestAnimationFrame(updateCount);
        } else {
            const finalValue = decimals > 0 ? end.toFixed(decimals) : end.toLocaleString();
            element.textContent = finalValue + suffix;
            element.classList.add('counter-animated');
        }
    }

    requestAnimationFrame(updateCount);
}
