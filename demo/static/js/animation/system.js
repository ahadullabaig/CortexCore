/**
 * PHASE 3: ANIMATION SYSTEMS
 * Centralized animation initialization and management
 */

import { detectDeviceCapabilities } from '../utils/device.js';
import { initializeScrollAnimations } from './scroll.js';
import { createFloatingParticles, pauseParticleAnimations, resumeParticleAnimations } from './particles.js';
import { setupPerformanceMonitoring } from './performance.js';

/**
 * Initialize all animation systems
 */
export function initializeAnimationSystems() {
    console.log('🎬 Initializing Phase 3 Animation Systems...');

    // 1. Detect device capabilities and set optimization class
    detectDeviceCapabilities();

    // 2. Initialize Intersection Observer for scroll-triggered animations
    initializeScrollAnimations();

    // 3. Setup button ripple effects
    setupButtonRipples();

    // 4. Create floating particles (if not low-end device)
    // MEMORY OPTIMIZATION: Reduced from 25 to 15 particles
    if (!document.body.classList.contains('low-end-device')) {
        createFloatingParticles(15); // 15 particles (reduced for performance)
    }

    // 5. Create grid scanner
    if (!document.body.classList.contains('low-end-device')) {
        createGridScanner();
    }

    // 6. Setup performance monitoring (dev mode)
    if (window.location.search.includes('debug')) {
        setupPerformanceMonitoring();
    }

    // 7. Pause animations when tab is inactive (battery saving)
    setupVisibilityHandler();

    console.log('✅ Animation systems initialized');
}

/**
 * Setup button ripple effects
 */
function setupButtonRipples() {
    const buttons = document.querySelectorAll('.btn');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Don't add ripple if button is disabled
            if (this.disabled) return;

            // Create ripple element
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');

            // Get click position relative to button
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Position ripple at click location
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';

            // Add ripple to button
            this.appendChild(ripple);

            // Remove ripple after animation completes (600ms)
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    console.log(`💫 Button ripple effects enabled for ${buttons.length} buttons`);
}

/**
 * Create grid scanner effect
 */
function createGridScanner() {
    // Create scanner if it doesn't exist
    if (!document.querySelector('.grid-scanner')) {
        const scanner = document.createElement('div');
        scanner.classList.add('grid-scanner');
        document.body.appendChild(scanner);
        console.log('📡 Grid scanner effect enabled');
    }
}

/**
 * Pause heavy animations when tab is hidden (battery saving)
 * MEMORY LEAK FIX: Target specific animations, not all children
 */
function setupVisibilityHandler() {
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            // Pause particle animations specifically
            pauseParticleAnimations();

            // Pause grid scanner
            const gridScanner = document.querySelector('.grid-scanner');
            if (gridScanner) {
                gridScanner.style.animationPlayState = 'paused';
            }

            console.log('⏸️ Heavy animations paused (tab inactive - saving battery)');
        } else {
            // Resume particle animations
            resumeParticleAnimations();

            // Resume grid scanner
            const gridScanner = document.querySelector('.grid-scanner');
            if (gridScanner) {
                gridScanner.style.animationPlayState = 'running';
            }

            console.log('▶️ Animations resumed (tab active)');
        }
    });
}
