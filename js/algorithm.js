(function () {
    const items = Array.from(document.querySelectorAll('.reveal-item'));
    if (!items.length) return;

    const staggerGroups = Array.from(document.querySelectorAll('.reveal-stagger'));
    staggerGroups.forEach((group) => {
        const children = Array.from(group.querySelectorAll(':scope > .reveal-item'));
        children.forEach((el, idx) => {
            el.style.transitionDelay = `${Math.min(idx, 8) * 70}ms`;
        });
    });

    const revealAll = () => items.forEach((el) => el.classList.add('is-visible'));

    if (!('IntersectionObserver' in window)) {
        revealAll();
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            });
        },
        {
            root: null,
            threshold: 0.12,
            rootMargin: '0px 0px -8% 0px',
        }
    );

    items.forEach((el) => observer.observe(el));
})();
