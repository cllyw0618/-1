/*!
* Start Bootstrap - Agency v7.0.12 (https://startbootstrap.com/theme/agency)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-agency/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    //  Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            rootMargin: '0px 0px -40%',
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

    const revealSelectors = [
        '.similar-hero-copy',
        '.similar-visual',
        '.similar-heading',
        '.similar-card',
        '.similar-compare-wrap',
        '.similar-highlight',
        '.similar-product-card',
        '.similar-scenario-section .app-card',
        '.similar-final-cta .summary-card',
        '.results-hero-content',
        '.results-visual',
        '.result-summary-bridge .summary-card',
        '.results-heading',
        '.results-table-wrap',
        '.competitor-card',
        '.ablation-insight-card',
        '.result-showcase-card',
        '.cluster-page .app-hero',
        '.cluster-page .scenario-section .section-header',
        '.cluster-page .scenario-stats-card',
        '.cluster-page .workspace-section .section-header',
        '.cluster-page .cluster-visual-stage',
        '.cluster-page .cluster-data-panel',
        '.cluster-page .app-overview .section-header',
        '.cluster-page .edge-list-card',
        '.cluster-page .app-deep .section-header',
        '.cluster-page .deep-accordion details',
        '.cluster-page .bottom-cta',
    ];

    const revealItems = Array.from(document.querySelectorAll(revealSelectors.join(',')));
    if (!revealItems.length) {
        return;
    }

    revealItems.forEach((item, index) => {
        item.classList.add('page-reveal');
        if (
            item.matches('.similar-visual, .results-visual, .summary-card, .results-table-wrap, .similar-compare-wrap')
        ) {
            item.classList.add('page-reveal-scale');
        }
        item.style.setProperty('--reveal-delay', `${Math.min(index % 6, 5) * 55}ms`);
    });

    const showAllRevealItems = () => {
        revealItems.forEach((item) => item.classList.add('is-visible'));
    };

    if (!('IntersectionObserver' in window)) {
        showAllRevealItems();
        return;
    }

    const revealObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                revealObserver.unobserve(entry.target);
            });
        },
        {
            threshold: 0.12,
            rootMargin: '0px 0px -8% 0px',
        }
    );

    revealItems.forEach((item) => revealObserver.observe(item));

});
