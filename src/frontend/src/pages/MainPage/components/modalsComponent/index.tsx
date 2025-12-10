// Modals.tsx
import TemplatesModal from "@/modals/templatesModal"; // Re-enable standard template modal
// import AIFlowAssistantModal from "@/modals/aiFlowAssistantModal"; // Keep available but unused
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
    {/* Use standard Templates modal for creating flows */}
    {openModal && <TemplatesModal open={openModal} setOpen={setOpenModal} />}
    {/* {openModal && <AIFlowAssistantModal open={openModal} setOpen={setOpenModal} />} */}
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
