:parenttoc: True

Examples
==========

.. toctree::
   :numbered:
   :maxdepth: 3

   Examples <./examples>

.. raw:: html
   
   <style>
  /* Border radius parameter */
  :root {
    --main-border-radius: 15px;
  }

  .light {
    --bg: white;
    --text-color: black;
    --accent: #1f4184;
    --header-number: #9bc5f1;
    --header-bg: #1f4184;
    --header-text-color: white;
    --section-number: #ddeeff;
    --under-section-number: #f1f8ff;
    --border: var(--accent);
  }

  .dark {
    --bg: #121212;
    --text-color: white;
    --accent: white;
    --header-number: #4c83c7;
    --header-bg: #1e1e1e;
    --header-text-color: white;
    --section-number: rgb(37, 37, 37);
    --under-section-number: rgb(22, 22, 22);
    --border: #1e1e1e;
    --color: #80afe9;
  }

  .toctree-wrapper > ul {
    display: grid;
    grid-gap: 15px;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    grid-auto-rows: 0px;
    padding-left: 0px;
  }

  .reference {
    color: var(--text-color);
  }

  .item-header {
    display: grid;
    grid-template-columns: 15% 85%;
    background-color: var(--header-bg);
    align-items: center;
    border-top-right-radius: var(--main-border-radius);
    border-top-left-radius: var(--main-border-radius);
  }

  .item-header-number {
    color: var(--header-number);
    text-align: center;
    font-size: 2em;
    font-weight: 700;
    margin: auto;
  }

  .item-header .reference {
    color: var(--header-text-color);
    font-size: 15px;
    font-weight: 700;
  }

  .toctree-wrapper .toctree-l1 {
    list-style: none;
    padding-bottom: 10px;
  }

  .toctree-wrapper .toctree-l1 ul {
    font-weight: 500;
    border-top-right-radius: 0px;
    border-top-left-radius: 0px;
    list-style-position: outside;
    padding-left: 0px;
  }

  .toctree-wrapper .toctree-l1 ul .toctree-l3 {
    font-weight: 400;
  }

  .toctree-wrapper li {
    list-style: none;
  }

  .toctree-wrapper .item {
    background-color: var(--bg);
    border-radius: var(--main-border-radius);
    border: 1px solid var(--border);
  }

  .item .toctree-l1 .toctree-l2 {
    display: grid;
    grid-template-columns: 15% 85%;
    align-items: center;
    padding-top: 0.5em;
    line-height: 1.1em;
  }

  .item .toctree-l1 .toctree-l2 .toctree-l3 {
    display: grid;
    grid-template-columns: 15% 85%;
    align-items: center;
    padding-top: 0.5em;
    line-height: 1.1em;
  }

  .item .toctree-l1 .toctree-l2 > ul {
    grid-column: 2;
  }

  .Section-number {
    background-color: var(--section-number);
    text-align: center;
    border-radius: 100px;
    border: 5px var(--bg) solid;
    font-weight: 500;
    margin: auto;
  }

  .under-Section-number {
    background-color: var(--under-section-number);
    border-radius: 100px;
    text-align: center;
    border: 5px var(--bg) solid;
    margin: auto;
  }
   </style>
   <script defer>
   // resizes grid item for masonry grid structure
   function resizeGridItem(item) {
      grid = document.querySelector("div.toctree-wrapper ul");
      rowHeight = parseInt(
         window.getComputedStyle(grid).getPropertyValue("grid-auto-rows")
      );
      rowGap = parseInt(
         window.getComputedStyle(grid).getPropertyValue("grid-row-gap")
      );
      rowSpan = Math.ceil(
         (item.querySelector(".toctree-l1").getBoundingClientRect().height +
         rowGap) /
         (rowHeight + rowGap)
      );
      item.style.gridRowEnd = "span " + rowSpan;
   }

   // loops through all grid items and resizes them
   function resizeAllGridItems() {
      allItems = document.getElementsByClassName("item");
      for (let x = 0; x < allItems.length; x++) {
         resizeGridItem(allItems[x]);
      }
   }

   // add chapter number to item-header
   function addChapters(nodelist) {
      for (let i = 0; i < nodelist.length; i++) {
         var newP = document.createElement("p");
         var textNode = document.createTextNode(
         (
            "0" + nodelist[i].children[0].children[0].textContent.split(".")[0]
         ).slice(-2)
         );
         newP.appendChild(textNode);
         newP.className = "item-header-number";
         nodelist[i].children[0].insertBefore(
         newP,
         nodelist[i].children[0].children[0]
         );
         nodelist[i].children[0].children[1].textContent = nodelist[
         i
         ].children[0].children[1].textContent
         .split(".")[1]
         .substring(1);
      }
   }

   // remove numbering of section string and add it back as p in li container
   function addSections(nodelist) {
      for (let i = 0; i < nodelist.length; i++) {
         for (let k = 0; k < nodelist[i].children.length; k++) {
         if (nodelist[i].children[k].classList.contains("toctree-l3")) {
            continue;
         }
         if (nodelist[i].children[k].querySelector("ul") !== null) {
            for (
               let n = 0;
               n < nodelist[i].children[k].querySelector("ul").children.length;
               n++
            ) {
               var newP = document.createElement("p");
               var underSectionNumber = nodelist[i].children[k]
               .querySelector("ul")
               .children[n].children[0].textContent.split(" ")[0];
               var underSectionName = nodelist[i].children[k]
               .querySelector("ul")
               .children[n].children[0].textContent.split(" ")
               .slice(1)
               .join(" ");
               var textNode = document.createTextNode(underSectionNumber);
               newP.appendChild(textNode);
               newP.className = "under-Section-number";
               nodelist[i].children[k]
               .querySelector("ul")
               .children[n].insertBefore(
                  newP,
                  nodelist[i].children[k].querySelector("ul").children[n]
                     .children[0]
               );
               nodelist[i].children[k].querySelector("ul").children[
               n
               ].children[1].textContent = underSectionName;
            }
         }
         var newP = document.createElement("p");
         var sectionNumber =
            nodelist[i].children[k].children[0].textContent.split(" ")[0];
         var sectionName = nodelist[i].children[k].children[0].textContent
            .split(" ")
            .slice(1)
            .join(" ");
         var textNode = document.createTextNode(sectionNumber);
         newP.appendChild(textNode);
         newP.className = "Section-number";
         nodelist[i].children[k].insertBefore(
            newP,
            nodelist[i].children[k].children[0]
         );
         nodelist[i].children[k].children[1].textContent = sectionName;
         }
      }
   }

   $(document).ready(function () {
      // put chapters in 'item' containers
      $(".toctree-l1").wrap("<div class='item'></div>");

      // add 'item-header' class to chapter headers
      var headerNodeList = document.querySelectorAll(
         "div.toctree-wrapper .item .toctree-l1"
      );
      for (let i = 0; i < headerNodeList.length; i++) {
         $(headerNodeList[i].children[0]).wrap("<div class='item-header'></div>");
      }
      // execute functions on load
      window.onload = addChapters(
         document.querySelectorAll("div.toctree-wrapper .item .toctree-l1")
      );
      window.onload = addSections(document.querySelectorAll(".toctree-l1 ul"));

      // generate masonry grid and maintain it
      window.onload = resizeAllGridItems();
      window.addEventListener("resize", resizeAllGridItems);
   });
   resizeAllGridItems();
   </script>
   <script defer>
   var observer = new MutationObserver(function (mutations) {
      const dark = document.documentElement.dataset.theme == "dark";
      var toc_wrapper = document.getElementsByClassName("toctree-wrapper")[0];
      if (dark) {
         toc_wrapper.classList.remove("light");
         toc_wrapper.classList.add("dark");
      } else {
         toc_wrapper.classList.add("light");
         toc_wrapper.classList.remove("dark");
      }
   });
   observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
   });
   setTimeout(function () {resizeAllGridItems();}, 500);
   </script>
