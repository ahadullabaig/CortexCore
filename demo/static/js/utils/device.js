/**
 * Device Capability Detection
 * Detects low-end devices and optimizes animations accordingly
 */

/**
 * Detect device capabilities and apply optimization class
 * Sets .low-end-device class on body if device has limited resources
 */
export function detectDeviceCapabilities() {
    const body = document.body;

    // Check if low-end device (limited memory, slow CPU, or mobile)
    const isLowEnd = (
        navigator.hardwareConcurrency <= 2 || // 2 or fewer CPU cores
        navigator.deviceMemory <= 4 || // 4GB or less RAM
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    );

    // Check battery status if available
    if (navigator.getBattery) {
        navigator.getBattery().then(battery => {
            // If battery is low (<20%) and not charging, disable heavy animations
            if (battery.level < 0.2 && !battery.charging) {
                body.classList.add('low-end-device');
                console.log('⚡ Low battery detected - disabling heavy animations');
            }
        });
    }

    if (isLowEnd) {
        body.classList.add('low-end-device');
        console.log('📱 Low-end device detected - optimizing animations');
    } else {
        console.log('💪 High-performance device detected - full animations enabled');
    }
}
