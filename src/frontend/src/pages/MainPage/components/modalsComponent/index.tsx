// Modals.tsx
// import TemplatesModal from "@/modals/templatesModal"; // Commented out - replaced with AI Assistant
import AIFlowAssistantModal from "@/modals/aiFlowAssistantModal";
import DeleteConfirmationModal from "../../../../modals/deleteConfirmationModal";

interface ModalsProps {
  openModal: boolean;
  setOpenModal: (value: boolean) => void;
  openDeleteFolderModal: boolean;
  setOpenDeleteFolderModal: (value: boolean) => void;
  handleDeleteFolder: () => void;
}

const ModalsComponent = ({
  openModal = false,
  setOpenModal = () => {},
  openDeleteFolderModal = false,
  setOpenDeleteFolderModal = () => {},
  handleDeleteFolder = () => {},
}: ModalsProps) => (
  <>
    {/* Replaced TemplatesModal with AI Flow Assistant */}
    {openModal && <AIFlowAssistantModal open={openModal} setOpen={setOpenModal} />}
    {/* {openModal && <TemplatesModal open={openModal} setOpen={setOpenModal} />} */}
    {openDeleteFolderModal && (
      <DeleteConfirmationModal
        open={openDeleteFolderModal}
        setOpen={setOpenDeleteFolderModal}
        onConfirm={() => {
          handleDeleteFolder();
          setOpenDeleteFolderModal(false);
        }}
        description="folder"
        note={"and all associated flows and components"}
      >
        <></>
      </DeleteConfirmationModal>
    )}
  </>
);

export default ModalsComponent;
