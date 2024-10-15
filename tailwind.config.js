/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      backgroundColor: {
        customBackground: "#28282B",
        customSidebarColor: "#c28a02",
        customSubOption: "#D6D6D6",
        customActiveOption: "#c28a02",
        customLightBackground: "#c28a02",
      },
    },
    fontFamily: {
      Arial: ["Arial"],
    },
    textColor: {
      ebony: "#EBEBEB",
      active: "#000000",
    },
    borderRadius: {
      customBtn: "20px 20px 20px 20px",
      customFile: "20px 0px 0px 20px",
      customImageDisplay: "24px",
      customCanvasDisplay: "15px",
      customPopUp: "16px 0px 16px 16px",
      customSidebar: "56px",
    },
    // Customize border color
    borderColor: {
      customBtn: "#c28a02",
      customFile: "#c28a02",
      customImageDisplay: "#FFF",
      customLightImageDisplay: "#071017",
      customLightBorder: "#F7F7F2",
    },
    textShadow: {
      md: "0px 0px 20px #755300",
    },
    boxShadow: {
      customShadow: "0px 8px 12.3px 0px #c28a02",
      customLightShadow: "0px 8px 12.3px 0px #c28a02",
      customImageDisplay: "0px 4px 20px 0px #c28a02",
      customImageLightDisplay: "0px 4px 20px 0px #c28a02",
    },
  },
  plugins: [],
};
