/**
 * Scroll-Triggered Animations (Intersection Observer)
 * Animates elements when they enter viewport
 */

/**
 * Initialize Intersection Observer for scroll animations
 */
export function initializeScrollAnimations() {
    // Create intersection observer
    const observerOptions = {
        root: null, // viewport
        rootMargin: '0px',
        threshold: 0.1 // Trigger when 10% of element is visible
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add 'animated' class to trigger animation
                entry.target.classList.add('animated');

                // Optional: Unobserve after animation (one-time animation)
                // observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all elements with .animate-on-scroll class
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach(el => observer.observe(el));

    console.log(`👁️ Observing ${animatedElements.length} elements for scroll animations`);
}
