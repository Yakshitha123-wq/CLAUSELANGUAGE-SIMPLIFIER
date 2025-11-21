// Universal Mouse Glow Effect for All Pages
document.addEventListener('DOMContentLoaded', function() {
    const body = document.body;
    
    // Mouse move event to create glow effect
    body.addEventListener('mousemove', (e) => {
        const x = e.clientX;
        const y = e.clientY;
        
        // Set CSS custom properties for mouse position
        body.style.setProperty('--mouse-x', x + 'px');
        body.style.setProperty('--mouse-y', y + 'px');
        
        // Add active class to show the glow
        body.classList.add('mouse-active');
        
        // Update background with radial gradient following cursor
        body.style.background = `
            radial-gradient(4px circle at ${x}px ${y}px, 
                rgba(178, 102, 255, 0.15), 
                transparent 40%),
            #ffffff
        `;
    });
    
    // Remove glow when mouse leaves the page
    body.addEventListener('mouseleave', () => {
        body.classList.remove('mouse-active');
        body.style.background = '#ffffff';
    });
});