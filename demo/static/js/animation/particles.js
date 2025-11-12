/**
 * Floating Particles System
 * Creates neural network-inspired particle effects
 */

import { animationState } from '../core/config.js';

/**
 * Create floating particles with proper lifecycle management
 * MEMORY LEAK FIX: Store particle references for cleanup
 * @param {number} count - Number of particles to create
 */
export function createFloatingParticles(count) {
    // Create container if it doesn't exist
    let container = document.querySelector('.neural-particles');

    if (!container) {
        container = document.createElement('div');
        container.classList.add('neural-particles');
        document.body.appendChild(container);
    }

    // Clear existing particles and their references
    clearFloatingParticles();

    // Create particles
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');

        // Randomize size
        const sizeRandom = Math.random();
        if (sizeRandom < 0.3) {
            particle.classList.add('small');
        } else if (sizeRandom > 0.7) {
            particle.classList.add('large');
        }

        // Random starting position
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';

        // Random animation duration (10-30 seconds)
        particle.style.animationDuration = (10 + Math.random() * 20) + 's';

        // Random animation delay for stagger effect
        particle.style.animationDelay = (Math.random() * 5) + 's';

        container.appendChild(particle);

        // Store reference for cleanup
        animationState.particles.push(particle);
    }

    console.log(`✨ Created ${count} floating particles (optimized for performance)`);
}

/**
 * Clear all floating particles and free memory
 */
export function clearFloatingParticles() {
    const container = document.querySelector('.neural-particles');

    if (container) {
        container.innerHTML = '';
    }

    // Clear particle references
    animationState.particles = [];
}

/**
 * Pause particle animations (called when tab is hidden)
 */
export function pauseParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'paused';
    });
}

/**
 * Resume particle animations (called when tab is visible)
 */
export function resumeParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'running';
    });
}
