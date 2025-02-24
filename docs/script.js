document.addEventListener("DOMContentLoaded", function () {
    const backToTop = document.getElementById("backToTop");
    backToTop.style.display = "none"; // Initially hide the button
    backToTop.style.position = "fixed";
    backToTop.style.bottom = "20px";
    backToTop.style.right = "20px";
    backToTop.style.padding = "10px 15px";
    backToTop.style.backgroundColor = "black";
    backToTop.style.color = "white";
    backToTop.style.borderRadius = "5px";
    backToTop.style.cursor = "pointer";
    
    document.querySelectorAll(".nav-menu a").forEach(anchor => {
        anchor.addEventListener("click", function (event) {
            event.preventDefault();
            const targetId = this.getAttribute("href").substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 50,
                    behavior: "smooth"
                });
                backToTop.style.display = "block";
            }
        });
    });

    backToTop.addEventListener("click", function (event) {
        event.preventDefault();
        window.scrollTo({ top: 0, behavior: "smooth" });
        backToTop.style.display = "none";
    });
});
